// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeMap;
use std::fs::File as StdFile;
use std::io::BufReader;
use std::io::Seek;
use std::io::SeekFrom;
use std::iter::once;
use std::path::Path;
use std::path::PathBuf;
use std::ptr::NonNull;
use std::sync::Arc;

use anyhow::Context;
use arrow_array::PrimitiveArray;
use arrow_array::types::Int64Type;
use arrow_buffer::Buffer;
use arrow_ipc::Block;
use arrow_ipc::convert::fb_to_schema;
use arrow_ipc::reader::FileDecoder;
use arrow_ipc::reader::read_footer_length;
use arrow_ipc::root_as_footer;
use arrow_select::concat::concat_batches;
use arrow_select::take::take_record_batch;
use async_trait::async_trait;
use bytes::Bytes;
use memmap2::Mmap;
use parquet::arrow::arrow_reader::ArrowReaderMetadata;
use parquet::arrow::arrow_reader::ArrowReaderOptions;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::arrow_reader::RowSelection;
use parquet::file::metadata::PageIndexPolicy;
use parquet::file::reader::ChunkReader;
use parquet::file::reader::Length;
use vortex::array::Canonical;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::stream::ArrayStreamExt;
use vortex::buffer::Buffer as VortexBuffer;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::VortexFile;
use vortex::io::std_file::read_exact_at;
use vortex::scan::strict_sorted_buffer::StrictSortedBuffer;

use crate::Format;
use crate::SESSION;
use crate::random_access::ARROW_ROW_OFFSETS_METADATA_KEY;
use crate::random_access::RandomAccessor;
use crate::random_access::RandomAccessorRet;

/// Slice the memory-mapped file down to one IPC block (message metadata plus body).
fn block_data(data: &Buffer, block: &Block) -> anyhow::Result<Buffer> {
    let offset = usize::try_from(block.offset())?;
    let len = usize::try_from(block.metaDataLength())?
        .checked_add(usize::try_from(block.bodyLength())?)
        .context("Arrow IPC block length overflow")?;
    anyhow::ensure!(
        offset.checked_add(len).is_some_and(|end| end <= data.len()),
        "Arrow IPC block lies outside the file"
    );
    Ok(data.slice_with_length(offset, len))
}

/// Random accessor for uncompressed Arrow IPC files.
///
/// The file is memory-mapped and decoded zero-copy, the way Arrow's reference readers do point
/// lookups: taking a row touches only the pages holding that row's bytes rather than copying the
/// whole record batch through a read syscall.
pub struct ArrowIpcRandomAccessor {
    name: String,
    /// The whole file, wrapped as an Arrow buffer that keeps the mapping alive.
    data: Buffer,
    decoder: FileDecoder,
    /// One block per record batch, in file order.
    blocks: Vec<Block>,
    row_offsets: Vec<u64>,
    schema: arrow_schema::SchemaRef,
}

impl ArrowIpcRandomAccessor {
    pub fn open(path: PathBuf, name: impl Into<String>) -> anyhow::Result<Self> {
        let file = StdFile::open(path)?;
        // SAFETY: the mapping is read-only and the benchmark data file is never modified while
        // an accessor holds it.
        let mmap = unsafe { Mmap::map(&file)? };
        let len = mmap.len();
        let ptr = NonNull::new(mmap.as_ptr().cast_mut()).context("Arrow IPC file is empty")?;
        // SAFETY: `ptr` and `len` describe exactly the mapping owned by `mmap`, which the buffer
        // keeps alive as its allocation.
        let data = unsafe { Buffer::from_custom_allocation(ptr, len, Arc::new(mmap)) };

        anyhow::ensure!(len >= 10, "Arrow IPC file is too small to hold a footer");
        let trailer_start = len - 10;
        let footer_len = read_footer_length(data[trailer_start..].try_into()?)?;
        let footer_start = trailer_start
            .checked_sub(footer_len)
            .context("Arrow IPC footer length exceeds the file")?;
        let footer = root_as_footer(&data[footer_start..trailer_start])?;
        let schema = Arc::new(fb_to_schema(
            footer.schema().context("Arrow IPC footer has no schema")?,
        ));

        let row_offsets = footer
            .custom_metadata()
            .into_iter()
            .flatten()
            .find(|kv| kv.key() == Some(ARROW_ROW_OFFSETS_METADATA_KEY))
            .and_then(|kv| kv.value())
            .context("Arrow IPC file is missing row-offset metadata")?
            .split(',')
            .map(str::parse)
            .collect::<Result<Vec<u64>, _>>()?;

        let mut decoder = FileDecoder::new(Arc::clone(&schema), footer.version());
        // SAFETY: the file was written by this benchmark's own converter and is read-only, so
        // the buffers it references are valid Arrow data. Arrow's reference IPC readers likewise
        // trust files they read, which keeps the decode zero-copy.
        decoder = unsafe { decoder.with_skip_validation(true) };
        for block in footer.dictionaries().into_iter().flatten() {
            decoder.read_dictionary(block, &block_data(&data, block)?)?;
        }
        let blocks: Vec<Block> = footer
            .recordBatches()
            .into_iter()
            .flatten()
            .copied()
            .collect();

        anyhow::ensure!(
            row_offsets.len() == blocks.len() + 1,
            "Arrow IPC row-offset metadata does not match the record batches"
        );
        anyhow::ensure!(
            row_offsets.first() == Some(&0),
            "Arrow IPC row offsets must start at zero"
        );
        anyhow::ensure!(
            row_offsets.windows(2).all(|window| window[0] <= window[1]),
            "Arrow IPC row offsets must be sorted"
        );
        Ok(Self {
            name: name.into(),
            data,
            decoder,
            blocks,
            row_offsets,
            schema,
        })
    }
}

#[async_trait]
impl RandomAccessor for ArrowIpcRandomAccessor {
    fn format(&self) -> Format {
        Format::ArrowIpc
    }

    fn name(&self) -> &str {
        &self.name
    }

    async fn take(&self, indices: &[u64]) -> anyhow::Result<RandomAccessorRet> {
        let mut by_batch = BTreeMap::<usize, Vec<i64>>::new();
        for &index in indices {
            let batch_idx = self.row_offsets.partition_point(|offset| *offset <= index);
            anyhow::ensure!(
                batch_idx > 0 && batch_idx < self.row_offsets.len(),
                "Arrow row index {index} is out of bounds"
            );
            let batch_idx = batch_idx - 1;
            by_batch
                .entry(batch_idx)
                .or_default()
                .push(i64::try_from(index - self.row_offsets[batch_idx])?);
        }

        let mut batches = Vec::with_capacity(by_batch.len());
        for (batch_idx, local_indices) in by_batch {
            let block = &self.blocks[batch_idx];
            let batch = self
                .decoder
                .read_record_batch(block, &block_data(&self.data, block)?)?
                .context("Arrow IPC record batch is missing")?;
            let indices = PrimitiveArray::<Int64Type>::from(local_indices);
            batches.push(take_record_batch(&batch, &indices)?);
        }

        Ok(RandomAccessorRet::RecordBatch(concat_batches(
            &self.schema,
            &batches,
        )?))
    }
}

/// Random accessor for Vortex format files.
///
/// The file handle is opened at construction time and reused across `take()` calls.
pub struct VortexRandomAccessor {
    name: String,
    format: Format,
    file: VortexFile,
}

impl VortexRandomAccessor {
    /// Open a Vortex file and return a ready-to-use accessor.
    pub async fn open(
        path: impl AsRef<Path>,
        name: impl Into<String>,
        format: Format,
    ) -> anyhow::Result<Self> {
        let file = SESSION
            .open_options()
            .with_layout_reader_cache()
            .open_path(path.as_ref())
            .await?;
        Ok(Self {
            name: name.into(),
            format,
            file,
        })
    }
}

#[async_trait]
impl RandomAccessor for VortexRandomAccessor {
    fn format(&self) -> Format {
        self.format
    }

    fn name(&self) -> &str {
        &self.name
    }

    async fn take(&self, indices: &[u64]) -> anyhow::Result<RandomAccessorRet> {
        let indices_buf: VortexBuffer<u64> = VortexBuffer::from(indices.to_vec());
        let array = self
            .file
            .scan()?
            .with_row_indices(StrictSortedBuffer::try_new(indices_buf)?)
            .into_array_stream()?
            .read_all()
            .await?;

        // We canonicalize / decompress for equivalence to Arrow's `RecordBatch`es.
        let mut ctx = SESSION.create_execution_ctx();
        let canonical = array.execute::<Canonical>(&mut ctx)?.into_array();
        Ok(RandomAccessorRet::ArrayRef(canonical))
    }
}

/// A Parquet [`ChunkReader`] that serves page reads with positional `pread` calls.
///
/// The `ChunkReader` impl for [`std::fs::File`] duplicates the descriptor and seeks for every
/// page. Vortex reads its segments with `pread` on a shared handle, so Parquet gets the same
/// syscall profile here.
#[derive(Clone)]
struct PreadFile {
    file: Arc<StdFile>,
    len: u64,
}

impl PreadFile {
    fn open(path: &Path) -> anyhow::Result<Self> {
        let file = StdFile::open(path)?;
        let len = file.metadata()?.len();
        Ok(Self {
            file: Arc::new(file),
            len,
        })
    }
}

impl Length for PreadFile {
    fn len(&self) -> u64 {
        self.len
    }
}

impl ChunkReader for PreadFile {
    type T = BufReader<StdFile>;

    fn get_read(&self, start: u64) -> parquet::errors::Result<Self::T> {
        let mut reader = self.file.try_clone()?;
        reader.seek(SeekFrom::Start(start))?;
        Ok(BufReader::new(reader))
    }

    fn get_bytes(&self, start: u64, length: usize) -> parquet::errors::Result<Bytes> {
        let mut buffer = vec![0u8; length];
        read_exact_at(&self.file, &mut buffer, start)?;
        Ok(buffer.into())
    }
}

/// Random accessor for Parquet format files.
///
/// The footer and page index are parsed once at construction. Each `take()` maps the indices to
/// row groups and hands the reader a [`RowSelection`], so the offset index lets it fetch and
/// decode only the pages holding the requested rows instead of the whole row group.
pub struct ParquetRandomAccessor {
    name: String,
    /// Cumulative row offsets per row group (length = num_row_groups + 1).
    row_group_offsets: Vec<u64>,
    /// Cached Arrow reader metadata (footer and page index) to avoid re-parsing on each take.
    arrow_metadata: ArrowReaderMetadata,
    file: PreadFile,
}

impl ParquetRandomAccessor {
    /// Open a Parquet file, parse the footer, and return a ready-to-use accessor.
    pub async fn open(path: PathBuf, name: impl Into<String>) -> anyhow::Result<Self> {
        let file = PreadFile::open(&path)?;
        let options = ArrowReaderOptions::new().with_page_index_policy(PageIndexPolicy::Required);
        let arrow_metadata = ArrowReaderMetadata::load(&file, options)?;

        let row_group_offsets = once(0u64)
            .chain(
                arrow_metadata
                    .metadata()
                    .row_groups()
                    .iter()
                    .map(|rg| u64::try_from(rg.num_rows()).unwrap_or_default()),
            )
            .scan(0u64, |acc, x| {
                *acc += x;
                Some(*acc)
            })
            .collect::<Vec<_>>();

        Ok(Self {
            name: name.into(),
            row_group_offsets,
            arrow_metadata,
            file,
        })
    }
}

#[async_trait]
impl RandomAccessor for ParquetRandomAccessor {
    fn format(&self) -> Format {
        Format::Parquet
    }

    fn name(&self) -> &str {
        &self.name
    }

    async fn take(&self, indices: &[u64]) -> anyhow::Result<RandomAccessorRet> {
        anyhow::ensure!(
            indices.windows(2).all(|window| window[0] < window[1]),
            "Parquet random access requires strictly increasing indices"
        );

        // Row selections are relative to the concatenation of the selected row groups, so track
        // where each selected row group starts within that concatenation.
        let mut row_groups = Vec::new();
        let mut ranges = Vec::with_capacity(indices.len());
        let mut selected_rows = 0usize;
        let mut current: Option<(usize, usize)> = None;
        for &index in indices {
            let row_group = self
                .row_group_offsets
                .partition_point(|offset| *offset <= index);
            anyhow::ensure!(
                row_group > 0 && row_group < self.row_group_offsets.len(),
                "Parquet row index {index} is out of bounds"
            );
            let row_group = row_group - 1;
            let base = match current {
                Some((rg, base)) if rg == row_group => base,
                _ => {
                    if let Some((prev, _)) = current {
                        selected_rows += self.row_group_len(prev)?;
                    }
                    current = Some((row_group, selected_rows));
                    row_groups.push(row_group);
                    selected_rows
                }
            };
            let local = usize::try_from(index - self.row_group_offsets[row_group])?;
            ranges.push(base + local..base + local + 1);
        }
        if let Some((last, _)) = current {
            selected_rows += self.row_group_len(last)?;
        }
        let selection = RowSelection::from_consecutive_ranges(ranges.into_iter(), selected_rows);

        let reader = ParquetRecordBatchReaderBuilder::new_with_metadata(
            self.file.clone(),
            self.arrow_metadata.clone(),
        )
        .with_row_groups(row_groups)
        .with_row_selection(selection)
        // Every selected row group yields one batch: the selection never exceeds the indices.
        .with_batch_size(indices.len().max(1))
        .build()?;

        let schema = Arc::clone(self.arrow_metadata.schema());
        let batches = reader.collect::<Result<Vec<_>, _>>()?;
        Ok(RandomAccessorRet::RecordBatch(concat_batches(
            &schema, &batches,
        )?))
    }
}

impl ParquetRandomAccessor {
    fn row_group_len(&self, row_group: usize) -> anyhow::Result<usize> {
        Ok(usize::try_from(
            self.row_group_offsets[row_group + 1] - self.row_group_offsets[row_group],
        )?)
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::Int64Array;
    use arrow_array::RecordBatch;
    use arrow_ipc::writer::FileWriter;
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::Schema;
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;

    use super::*;

    #[tokio::test]
    async fn arrow_ipc_random_accessor_takes_rows_across_record_batches() -> anyhow::Result<()> {
        let file = tempfile::NamedTempFile::new()?;
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
        {
            let mut writer = FileWriter::try_new(file.reopen()?, schema.as_ref())?;
            writer.write(&RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(Int64Array::from(vec![0, 1]))],
            )?)?;
            writer.write(&RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(Int64Array::from(vec![2, 3, 4]))],
            )?)?;
            writer.write_metadata(ARROW_ROW_OFFSETS_METADATA_KEY, "0,2,5");
            writer.finish()?;
        }

        let accessor = ArrowIpcRandomAccessor::open(file.path().to_path_buf(), "arrow-ipc")?;
        let RandomAccessorRet::RecordBatch(actual) = accessor.take(&[1, 3, 4]).await? else {
            anyhow::bail!("Arrow accessor returned a Vortex array")
        };
        let expected =
            RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![1, 3, 4]))])?;
        assert_eq!(actual, expected);
        Ok(())
    }

    #[tokio::test]
    async fn parquet_random_accessor_selects_rows_across_row_groups() -> anyhow::Result<()> {
        let file = tempfile::NamedTempFile::new()?;
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
        {
            // Three-row row groups with a one-row page each, so a lookup must skip pages and
            // row groups to land on the right rows.
            let props = WriterProperties::builder()
                .set_max_row_group_row_count(Some(3))
                .set_write_batch_size(1)
                .set_data_page_row_count_limit(1)
                .build();
            let mut writer =
                ArrowWriter::try_new(file.reopen()?, Arc::clone(&schema), Some(props))?;
            writer.write(&RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(Int64Array::from_iter_values(0..10))],
            )?)?;
            writer.close()?;
        }

        let accessor = ParquetRandomAccessor::open(file.path().to_path_buf(), "parquet").await?;
        assert_eq!(accessor.row_group_offsets, vec![0, 3, 6, 9, 10]);
        let RandomAccessorRet::RecordBatch(actual) = accessor.take(&[1, 2, 7, 9]).await? else {
            anyhow::bail!("Parquet accessor returned a Vortex array")
        };
        let expected =
            RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![1, 2, 7, 9]))])?;
        assert_eq!(actual, expected);
        Ok(())
    }
}
