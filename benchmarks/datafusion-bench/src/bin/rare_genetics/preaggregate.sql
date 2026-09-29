-- SPDX-License-Identifier: Apache-2.0
-- SPDX-FileCopyrightText: Copyright the Vortex contributors

WITH per_variant AS (
    SELECT contig, position,
           list_sum(array_transform(list_filter(samples, s -> s['is_case']), s -> s['genotype'])) AS cases,
           list_sum(array_transform(list_filter(samples, s -> NOT s['is_case']), s -> s['genotype'])) AS controls
    FROM variant_sample_matrix
)
SELECT g.gene_name,
       SUM(v.cases) AS n_case_mutations,
       SUM(v.controls) AS n_control_mutations
FROM per_variant v
JOIN genes g ON v.contig = g.contig
            AND v.position >= g.start_position
            AND v.position < g.end_position
GROUP BY g.gene_name
ORDER BY g.gene_name
