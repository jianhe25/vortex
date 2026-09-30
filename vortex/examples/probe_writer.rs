// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Probe: how many leaf chunks does the default writer produce for a comment-like string
//! column, depending on the size of the input batches?

use std::time::Instant;

use vortex::VortexSessionDefault;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::ChunkedArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::VarBinViewArray;
use vortex::buffer::ByteBufferMut;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::WriteOptionsSessionExt;
use vortex::layout::LayoutRef;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;

fn comment(i: usize) -> String {
    // ~27 bytes like l_comment, all distinct so the dict strategy does not fire.
    let words = ["carefully", "final", "deposits", "haggle", "blithely", "quick", "furious", "pending"];
    let a = words[i % 8];
    let b = words[(i / 8) % 8];
    let c = words[(i / 64) % 8];
    format!("{a} {b} {c} {i:08}")
}

fn column(n: usize) -> ArrayRef {
    VarBinViewArray::from_iter_str((0..n).map(comment)).into_array()
}

fn count_leaves(layout: &LayoutRef, depth: usize, counts: &mut Vec<(String, usize)>) {
    let name = format!("{:?}", layout.encoding_id());
    let children = layout.children().unwrap();
    if children.is_empty() {
        counts.push((format!("{}{name} rows={}", " ".repeat(depth), layout.row_count()), 1));
    }
    for child in children {
        count_leaves(&child, depth + 1, counts);
    }
}

fn run(session: &VortexSession, label: &str, chunks: Vec<ArrayRef>, dtype_len: usize) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let chunked = ChunkedArray::from_iter(chunks).into_array();
    let st = StructArray::from_fields(&[("comment", chunked)])
        .unwrap()
        .into_array();
    let mut buf = ByteBufferMut::empty();
    let start = Instant::now();
    rt.block_on(session.write_options().write(&mut buf, st.to_array_stream()))
        .unwrap();
    let elapsed = start.elapsed();
    let file = session.open_options().open_buffer(buf).unwrap();
    let layout = file.footer().layout().clone();
    println!("== {label}: {dtype_len} rows, wrote in {elapsed:?}");
    println!("{}", layout.display_tree());
    let mut counts = Vec::new();
    count_leaves(&layout, 0, &mut counts);
    println!("leaf layouts: {}", counts.len());
}

fn main() {
    let session = VortexSession::default().with_tokio();
    let n = 8192 * 64;
    let big = column(n);
    run(&session, "one 524288-row input batch", vec![big.clone()], n);
    let small: Vec<ArrayRef> = (0..64).map(|i| big.slice(i * 8192..(i + 1) * 8192).unwrap()).collect();
    run(&session, "64 x 8192-row slices of that batch", small, n);
    let fresh: Vec<ArrayRef> = (0..64).map(|i| column_range(i * 8192, (i + 1) * 8192)).collect();
    run(&session, "64 x 8192-row independently built batches", fresh, n);
}

fn column_range(start: usize, end: usize) -> ArrayRef {
    VarBinViewArray::from_iter_str((start..end).map(comment)).into_array()
}
