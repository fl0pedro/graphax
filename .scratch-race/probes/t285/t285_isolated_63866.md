## Table A — stored elements, growing broadcasts, dots

| case | optimum stored | mode | stored | val.shape | growing calls/elements | jaxpr growing | dots | temp B | dims |
|---|---|---|---|---|---|---|---|---|---|
| single_implicit_sparse | 60 | incumbent | 60 | (4, 3, 5) | 1/30 | 1/30 | 1 | 160 | SS |
| single_implicit_sparse | 60 | lazy off | 60 | (4, 3, 5) | 1/30 | 1/30 | 1 | 160 | SS |
| single_implicit_sparse | 60 | lazy nodemote | 60 | (4, 3, 5) | 1/30 | 1/30 | 1 | 160 | SS |
| single_implicit_sparse | 60 | lazy full | 60 | (4, 3, 5) | 0/0 | 0/0 | 1 | 0 | SS |
| single_implicit_sparse | 60 | planner | 60 | (4, 3, 5) | 0/0 | 0/0 | 1 | 0 | SS |
| single_implicit_sparse_rhs | 60 | incumbent | 60 | (4, 3, 5) | 1/18 | 1/18 | 1 | 96 | SS |
| single_implicit_sparse_rhs | 60 | lazy off | 60 | (4, 3, 5) | 1/18 | 1/18 | 1 | 96 | SS |
| single_implicit_sparse_rhs | 60 | lazy nodemote | 60 | (4, 3, 5) | 1/18 | 1/18 | 1 | 96 | SS |
| single_implicit_sparse_rhs | 60 | lazy full | 60 | (4, 3, 5) | 0/0 | 0/0 | 1 | 240 | SS |
| single_implicit_sparse_rhs | 60 | planner | 60 | (4, 3, 5) | 0/0 | 0/0 | 1 | 240 | SS |
| single_implicit_dense_block | 20 | incumbent | 60 | (4, 3, 5) | 1/16 | 1/16 | 1 | 96 | SS |
| single_implicit_dense_block | 20 | lazy off | 60 | (4, 3, 5) | 1/16 | 1/16 | 1 | 96 | SS |
| single_implicit_dense_block | 20 | lazy nodemote | 20 | (4, 5) | 0/0 | 0/0 | 1 | 0 | SS |
| single_implicit_dense_block | 20 | lazy full | 20 | (4, 5) | 0/0 | 0/0 | 1 | 0 | SS |
| single_implicit_dense_block | 20 | planner | 60 | (4, 3, 5) | 0/0 | 1/40 | 1 | 80 | SS |
| single_implicit_dense_contracted | 30 | incumbent | 30 | (2, 3, 5) | 1/18 | 1/18 | 1 | 288 | DDD |
| single_implicit_dense_contracted | 30 | lazy off | 30 | (2, 3, 5) | 1/18 | 1/18 | 1 | 288 | DDD |
| single_implicit_dense_contracted | 30 | lazy nodemote | 30 | (2, 3, 5) | 0/0 | 0/0 | 0 | 40 | DDD |
| single_implicit_dense_contracted | 30 | lazy full | 30 | (2, 3, 5) | 0/0 | 0/0 | 0 | 40 | DDD |
| single_implicit_dense_contracted | 30 | planner | 30 | (2, 3, 5) | 0/0 | 0/0 | 0 | 40 | DDD |
| single_implicit_dense_carried | 30 | incumbent | 30 | (2, 3, 5) | 1/12 | 1/12 | 1 | 288 | DDD |
| single_implicit_dense_carried | 30 | lazy off | 30 | (2, 3, 5) | 1/12 | 1/12 | 1 | 288 | DDD |
| single_implicit_dense_carried | 30 | lazy nodemote | 30 | (2, 3, 5) | 1/12 | 1/12 | 1 | 288 | DDD |
| single_implicit_dense_carried | 30 | lazy full | 30 | (2, 3, 5) | 0/0 | 0/0 | 1 | 312 | DDD |
| single_implicit_dense_carried | 30 | planner | 30 | (2, 3, 5) | 0/0 | 0/0 | 1 | 312 | DDD |
| double_implicit_contracted | 30 | incumbent | 30 | (2, 3, 5) | 0/0 | 0/0 | 0 | 0 | DDD |
| double_implicit_contracted | 30 | lazy off | 30 | (2, 3, 5) | 0/0 | 0/0 | 0 | 0 | DDD |
| double_implicit_contracted | 30 | lazy nodemote | 30 | (2, 3, 5) | 0/0 | 0/0 | 0 | 0 | DDD |
| double_implicit_contracted | 30 | lazy full | 30 | (2, 3, 5) | 0/0 | 0/0 | 0 | 0 | DDD |
| double_implicit_contracted | 30 | planner | 30 | (2, 3, 5) | 0/0 | 0/0 | 0 | 0 | DDD |
| double_implicit_batch | 15 | incumbent | 15 | (3, 5) | 2/32 | 2/32 | 1 | 416 | IDD |
| double_implicit_batch | 15 | lazy off | 15 | (3, 5) | 2/32 | 2/32 | 1 | 416 | IDD |
| double_implicit_batch | 15 | lazy nodemote | 15 | (3, 5) | 0/0 | 0/0 | 1 | 48 | IDD |
| double_implicit_batch | 15 | lazy full | 15 | (3, 5) | 0/0 | 0/0 | 1 | 48 | IDD |
| double_implicit_batch | 15 | planner | 15 | (3, 5) | 0/0 | 0/0 | 1 | 0 | IDD |
| uniform_operand | 10 | incumbent | 10 | (2, 5) | 1/23 | 1/23 | 1 | 416 | DID |
| uniform_operand | 10 | lazy off | 10 | (2, 5) | 1/23 | 1/23 | 1 | 416 | DID |
| uniform_operand | 10 | lazy nodemote | 10 | (2, 5) | 1/1 | 1/1 | 0 | 0 | DID |
| uniform_operand | 10 | lazy full | 10 | (2, 5) | 0/0 | 0/0 | 0 | 0 | DID |
| uniform_operand | 10 | planner | 10 | (2, 5) | 0/0 | 0/0 | 0 | 0 | DID |
| lcm_grid | 280 | incumbent | 280 | (4, 2, 5, 7) | 0/0 | 3/2087 | 1 | 3408 | SS |
| lcm_grid | 280 | lazy off | 280 | (4, 2, 5, 7) | 0/0 | 3/2087 | 1 | 3408 | SS |
| lcm_grid | 280 | lazy nodemote | 280 | (4, 2, 5, 7) | 0/0 | 3/2087 | 1 | 3408 | SS |
| lcm_grid | 280 | lazy full | 280 | (4, 2, 5, 7) | 0/0 | 3/2087 | 1 | 3408 | SS |
| lcm_grid | 280 | planner | 840 | (20, 42) | 0/0 | 4/1342 | 1 | 3376 | DD |
| spatial_sparse | 48 | incumbent | 48 | (4, 3, 4) | 1/40 | 1/40 | 1 | 704 | DSSD |
| spatial_sparse | 48 | lazy off | 48 | (4, 3, 4) | 1/40 | 1/40 | 1 | 704 | DSSD |
| spatial_sparse | 48 | lazy nodemote | 48 | (4, 3, 4) | 1/40 | 1/40 | 1 | 704 | DSSD |
| spatial_sparse | 48 | lazy full | 48 | (4, 3, 4) | 1/40 | 1/40 | 1 | 704 | DSSD |
| spatial_sparse | 48 | planner | 48 | (4, 3, 4) | 0/0 | 0/0 | 1 | 0 | DSSD |
| no_implicit_control | 60 | incumbent | 60 | (4, 3, 5) | 0/0 | 0/0 | 1 | 0 | SS |
| no_implicit_control | 60 | lazy off | 60 | (4, 3, 5) | 0/0 | 0/0 | 1 | 0 | SS |
| no_implicit_control | 60 | lazy nodemote | 60 | (4, 3, 5) | 0/0 | 0/0 | 1 | 0 | SS |
| no_implicit_control | 60 | lazy full | 60 | (4, 3, 5) | 0/0 | 0/0 | 1 | 0 | SS |
| no_implicit_control | 60 | planner | 60 | (4, 3, 5) | 0/0 | 0/0 | 1 | 0 | SS |

## Table B — which axis grew

| case | mode | grown axes (side.slot[pairing] x factor) | raw before -> target |
|---|---|---|---|
| single_implicit_sparse | incumbent | lhs.meta[contract]x4 | (1, 1, 2, 5)->(4, 1, 2, 5) |
| single_implicit_sparse | lazy off | lhs.meta[contract]x4 | (1, 1, 2, 5)->(4, 1, 2, 5) |
| single_implicit_sparse | lazy nodemote | lhs.meta[contract]x4 | (1, 1, 2, 5)->(4, 1, 2, 5) |
| single_implicit_sparse_rhs | incumbent | lhs.meta[contract]x4 | (1, 1, 3, 2)->(4, 1, 3, 2) |
| single_implicit_sparse_rhs | lazy off | lhs.meta[contract]x4 | (1, 1, 3, 2)->(4, 1, 3, 2) |
| single_implicit_sparse_rhs | lazy nodemote | lhs.meta[contract]x4 | (1, 1, 3, 2)->(4, 1, 3, 2) |
| single_implicit_dense_block | incumbent | lhs.block[contract]x3 | (4, 1, 1, 2)->(4, 1, 3, 2) |
| single_implicit_dense_block | lazy off | lhs.block[contract]x3 | (4, 1, 1, 2)->(4, 1, 3, 2) |
| single_implicit_dense_contracted | incumbent | lhs.split[contract]x4 | (1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1)->(1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 3, 1, 4, 1, 1, 1) |
| single_implicit_dense_contracted | lazy off | lhs.split[contract]x4 | (1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1)->(1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 3, 1, 4, 1, 1, 1) |
| single_implicit_dense_carried | incumbent | lhs.meta[batch_out]x2 | (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 4, 1, 1, 1)->(1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 3, 1, 4, 1, 1, 1) |
| single_implicit_dense_carried | lazy off | lhs.meta[batch_out]x2 | (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 4, 1, 1, 1)->(1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 3, 1, 4, 1, 1, 1) |
| single_implicit_dense_carried | lazy nodemote | lhs.meta[batch_out]x2 | (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 4, 1, 1, 1)->(1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 3, 1, 4, 1, 1, 1) |
| double_implicit_batch | incumbent | lhs.meta[batch_out]x4 | (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 4, 1, 1, 1)->(1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 3, 1, 4, 1, 1, 1); (1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 5)->(1, 1, 2, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 5) |
| double_implicit_batch | lazy off | lhs.meta[batch_out]x4 | (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 4, 1, 1, 1)->(1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 3, 1, 4, 1, 1, 1); (1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 5)->(1, 1, 2, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 5) |
| uniform_operand | incumbent | lhs.meta[batch_out]x2 lhs.block[spatial_out_lhs]x3 lhs.split[contract]x4 | (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)->(1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 3, 1, 4, 1, 1, 1) |
| uniform_operand | lazy off | lhs.meta[batch_out]x2 lhs.block[spatial_out_lhs]x3 lhs.split[contract]x4 | (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)->(1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 3, 1, 4, 1, 1, 1) |
| uniform_operand | lazy nodemote | lhs.meta[batch_out]x2 | (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)->(1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1) |
| spatial_sparse | incumbent | lhs.meta[spatial_sparse_lhs]x3 | (1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 1, 1, 4)->(1, 1, 1, 1, 1, 3, 1, 1, 5, 1, 1, 1, 1, 1, 1, 4) |
| spatial_sparse | lazy off | lhs.meta[spatial_sparse_lhs]x3 | (1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 1, 1, 4)->(1, 1, 1, 1, 1, 3, 1, 1, 5, 1, 1, 1, 1, 1, 1, 4) |
| spatial_sparse | lazy nodemote | lhs.meta[spatial_sparse_lhs]x3 | (1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 1, 1, 4)->(1, 1, 1, 1, 1, 3, 1, 1, 5, 1, 1, 1, 1, 1, 1, 4) |
| spatial_sparse | lazy full | lhs.meta[spatial_sparse_lhs]x3 | (1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 1, 1, 4)->(1, 1, 1, 1, 1, 3, 1, 1, 5, 1, 1, 1, 1, 1, 1, 4) |

## Table C — CPU HLO kernel census

| case | mode | dot shapes | reduce | copy | transpose | top-level broadcast | fusion kinds |
|---|---|---|---|---|---|---|---|
| single_implicit_sparse | incumbent | f32[4,3,5]{2,1,0} | 0 | 0 | 0 | - | {'kLoop': 1} |
| single_implicit_sparse | lazy off | f32[4,3,5]{2,1,0} | 0 | 0 | 0 | - | {'kLoop': 1} |
| single_implicit_sparse | lazy nodemote | f32[4,3,5]{2,1,0} | 0 | 0 | 0 | - | {'kLoop': 1} |
| single_implicit_sparse | lazy full | f32[12,5]{1,0} | 0 | 0 | 0 | - | {'kCustom': 1} |
| single_implicit_sparse | planner | f32[12,5]{1,0} | 0 | 0 | 0 | - | {'kCustom': 1} |
| single_implicit_sparse_rhs | incumbent | f32[4,3,5]{2,1,0} | 0 | 0 | 0 | - | {'kLoop': 1} |
| single_implicit_sparse_rhs | lazy off | f32[4,3,5]{2,1,0} | 0 | 0 | 0 | - | {'kLoop': 1} |
| single_implicit_sparse_rhs | lazy nodemote | f32[4,3,5]{2,1,0} | 0 | 0 | 0 | - | {'kLoop': 1} |
| single_implicit_sparse_rhs | lazy full | f32[3,20]{1,0} | 0 | 2 | 2 | - | {'kLoop': 2, 'kCustom': 1} |
| single_implicit_sparse_rhs | planner | f32[20,3]{1,0} | 0 | 2 | 2 | - | {'kLoop': 2, 'kCustom': 1} |
| single_implicit_dense_block | incumbent | f32[4,3,5]{2,1,0} | 0 | 0 | 0 | - | {'kLoop': 1} |
| single_implicit_dense_block | lazy off | f32[4,3,5]{2,1,0} | 0 | 0 | 0 | - | {'kLoop': 1} |
| single_implicit_dense_block | lazy nodemote | f32[4,5]{1,0} | 0 | 0 | 0 | - | - |
| single_implicit_dense_block | lazy full | f32[4,5]{1,0} | 0 | 0 | 0 | - | - |
| single_implicit_dense_block | planner | f32[4,5]{1,0} | 0 | 0 | 0 | - | {'kLoop': 1} |
| single_implicit_dense_contracted | incumbent | f32[2,3,5]{2,1,0} | 0 | 2 | 2 | - | {'kLoop': 2} |
| single_implicit_dense_contracted | lazy off | f32[2,3,5]{2,1,0} | 0 | 2 | 2 | - | {'kLoop': 2} |
| single_implicit_dense_contracted | lazy nodemote | - | 1 | 1 | 1 | - | {'kLoop': 2} |
| single_implicit_dense_contracted | lazy full | - | 1 | 1 | 1 | - | {'kLoop': 2} |
| single_implicit_dense_contracted | planner | - | 1 | 0 | 0 | - | {'kLoop': 2} |
| single_implicit_dense_carried | incumbent | f32[2,3,5]{2,1,0} | 0 | 4 | 4 | - | {'kLoop': 2} |
| single_implicit_dense_carried | lazy off | f32[2,3,5]{2,1,0} | 0 | 4 | 4 | - | {'kLoop': 2} |
| single_implicit_dense_carried | lazy nodemote | f32[2,3,5]{2,1,0} | 0 | 4 | 4 | - | {'kLoop': 2} |
| single_implicit_dense_carried | lazy full | f32[3,10]{1,0} | 0 | 7 | 6 | - | {'kLoop': 3, 'kCustom': 1} |
| single_implicit_dense_carried | planner | f32[10,3]{1,0} | 0 | 2 | 2 | - | {'kLoop': 2, 'kCustom': 1} |
| double_implicit_contracted | incumbent | - | 0 | 0 | 0 | - | {'kLoop': 1} |
| double_implicit_contracted | lazy off | - | 0 | 0 | 0 | - | {'kLoop': 1} |
| double_implicit_contracted | lazy nodemote | - | 0 | 0 | 0 | - | {'kLoop': 1} |
| double_implicit_contracted | lazy full | - | 0 | 0 | 0 | - | {'kLoop': 1} |
| double_implicit_contracted | planner | - | 0 | 0 | 0 | - | {'kLoop': 1} |
| double_implicit_batch | incumbent | f32[2,3,5]{2,1,0} | 0 | 3 | 2 | - | {'kLoop': 3} |
| double_implicit_batch | lazy off | f32[2,3,5]{2,1,0} | 0 | 3 | 2 | - | {'kLoop': 3} |
| double_implicit_batch | lazy nodemote | f32[3,5]{1,0} | 0 | 2 | 2 | - | {'kLoop': 1} |
| double_implicit_batch | lazy full | f32[3,5]{1,0} | 0 | 2 | 2 | - | {'kLoop': 1} |
| double_implicit_batch | planner | f32[3,5]{1,0} | 0 | 0 | 0 | - | - |
| uniform_operand | incumbent | f32[2,3,5]{2,1,0} | 0 | 3 | 2 | - | {'kLoop': 3} |
| uniform_operand | lazy off | f32[2,3,5]{2,1,0} | 0 | 3 | 2 | - | {'kLoop': 3} |
| uniform_operand | lazy nodemote | - | 1 | 1 | 1 | - | {'kLoop': 1} |
| uniform_operand | lazy full | - | 1 | 1 | 1 | - | {'kLoop': 1} |
| uniform_operand | planner | - | 1 | 0 | 0 | - | {'kLoop': 1} |
| lcm_grid | incumbent | f32[12,35]{1,0} | 1 | 2 | 2 | - | {'kLoop': 2, 'kCustom': 1} |
| lcm_grid | lazy off | f32[12,35]{1,0} | 1 | 2 | 2 | - | {'kLoop': 2, 'kCustom': 1} |
| lcm_grid | lazy nodemote | f32[12,35]{1,0} | 1 | 2 | 2 | - | {'kLoop': 2, 'kCustom': 1} |
| lcm_grid | lazy full | f32[12,35]{1,0} | 1 | 2 | 2 | - | {'kLoop': 2, 'kCustom': 1} |
| lcm_grid | planner | f32[20,42]{1,0} | 0 | 0 | 0 | - | {'kLoop': 8, 'kCustom': 1} |
| spatial_sparse | incumbent | f32[3,4,4]{2,1,0} | 0 | 4 | 3 | - | {'kLoop': 3} |
| spatial_sparse | lazy off | f32[3,4,4]{2,1,0} | 0 | 4 | 3 | - | {'kLoop': 3} |
| spatial_sparse | lazy nodemote | f32[3,4,4]{2,1,0} | 0 | 4 | 3 | - | {'kLoop': 3} |
| spatial_sparse | lazy full | f32[3,4,4]{2,1,0} | 0 | 4 | 3 | - | {'kLoop': 3} |
| spatial_sparse | planner | f32[12,4]{1,0} | 0 | 0 | 0 | - | {'kCustom': 1} |
| no_implicit_control | incumbent | f32[4,3,5]{2,1,0} | 0 | 0 | 0 | - | - |
| no_implicit_control | lazy off | f32[4,3,5]{2,1,0} | 0 | 0 | 0 | - | - |
| no_implicit_control | lazy nodemote | f32[4,3,5]{2,1,0} | 0 | 0 | 0 | - | - |
| no_implicit_control | lazy full | f32[4,3,5]{2,1,0} | 0 | 0 | 0 | - | - |
| no_implicit_control | planner | f32[4,3,5]{2,1,0} | 0 | 0 | 0 | - | - |

## Table D — the frame decision per pair

| case | mode | pairing | ol/orr | T/G | aligned | m_l/m_r | meta_lazy | lhs_lazy | rhs_lazy | demote |
|---|---|---|---|---|---|---|---|---|---|---|
| single_implicit_sparse | lazy off | contract | 4/4 | 4/4 | True | 4/1 | False | False | False | None |
| single_implicit_sparse | lazy nodemote | contract | 4/4 | 4/4 | True | 4/1 | False | False | False | None |
| single_implicit_sparse | lazy full | contract | 4/4 | 4/4 | True | 4/1 | False | False | False | r |
| single_implicit_sparse_rhs | lazy off | contract | 4/4 | 4/4 | True | 1/4 | False | False | False | None |
| single_implicit_sparse_rhs | lazy nodemote | contract | 4/4 | 4/4 | True | 1/4 | False | False | False | None |
| single_implicit_sparse_rhs | lazy full | contract | 4/4 | 4/4 | True | 1/4 | False | False | False | l |
| single_implicit_dense_block | lazy off | contract | 4/4 | 4/4 | True | 4/4 | False | False | False | None |
| single_implicit_dense_block | lazy nodemote | contract | 4/4 | 4/4 | True | 4/4 | False | True | False | None |
| single_implicit_dense_block | lazy full | contract | 4/4 | 4/4 | True | 4/4 | False | True | False | None |
| single_implicit_dense_contracted | lazy off | contract | 1/1 | 1/1 | True | 1/1 | False | False | False | None |
| single_implicit_dense_contracted | lazy off | batch_out | 2/2 | 2/2 | True | 2/2 | False | False | False | None |
| single_implicit_dense_contracted | lazy off | spatial_out_lhs | 1/1 | 1/1 | True | 1/1 | False | False | False | None |
| single_implicit_dense_contracted | lazy off | spatial_primal_rhs | 1/1 | 1/1 | True | 1/1 | False | False | False | None |
| single_implicit_dense_contracted | lazy nodemote | contract | 1/1 | 1/1 | True | 1/1 | False | False | False | None |
| single_implicit_dense_contracted | lazy nodemote | batch_out | 2/2 | 2/2 | True | 2/2 | False | False | False | None |
| single_implicit_dense_contracted | lazy nodemote | spatial_out_lhs | 1/1 | 1/1 | True | 1/1 | False | False | False | None |
| single_implicit_dense_contracted | lazy nodemote | spatial_primal_rhs | 1/1 | 1/1 | True | 1/1 | False | False | False | None |
| single_implicit_dense_contracted | lazy full | contract | 1/1 | 1/1 | True | 1/1 | False | False | False | None |
| single_implicit_dense_contracted | lazy full | batch_out | 2/2 | 2/2 | True | 2/2 | False | False | False | None |
| single_implicit_dense_contracted | lazy full | spatial_out_lhs | 1/1 | 1/1 | True | 1/1 | False | False | False | None |
| single_implicit_dense_contracted | lazy full | spatial_primal_rhs | 1/1 | 1/1 | True | 1/1 | False | False | False | None |
| single_implicit_dense_carried | lazy off | contract | 1/1 | 1/1 | True | 1/1 | False | False | False | None |
| single_implicit_dense_carried | lazy off | batch_out | 2/2 | 2/2 | True | 1/2 | False | False | False | None |
| single_implicit_dense_carried | lazy off | spatial_out_lhs | 1/1 | 1/1 | True | 1/1 | False | False | False | None |
| single_implicit_dense_carried | lazy off | spatial_primal_rhs | 1/1 | 1/1 | True | 1/1 | False | False | False | None |
| single_implicit_dense_carried | lazy nodemote | contract | 1/1 | 1/1 | True | 1/1 | False | False | False | None |
| single_implicit_dense_carried | lazy nodemote | batch_out | 2/2 | 2/2 | True | 1/2 | False | False | False | None |
| single_implicit_dense_carried | lazy nodemote | spatial_out_lhs | 1/1 | 1/1 | True | 1/1 | False | False | False | None |
| single_implicit_dense_carried | lazy nodemote | spatial_primal_rhs | 1/1 | 1/1 | True | 1/1 | False | False | False | None |
| single_implicit_dense_carried | lazy full | contract | 1/1 | 1/1 | True | 1/1 | False | False | False | None |
| single_implicit_dense_carried | lazy full | batch_out | 2/2 | 2/2 | True | 1/2 | False | False | False | l |
| single_implicit_dense_carried | lazy full | spatial_out_lhs | 1/1 | 1/1 | True | 1/1 | False | False | False | None |
| single_implicit_dense_carried | lazy full | spatial_primal_rhs | 1/1 | 1/1 | True | 1/1 | False | False | False | None |
| double_implicit_contracted | lazy off | batch_out | 2/2 | 2/2 | True | 2/2 | False | False | False | None |
| double_implicit_contracted | lazy off | spatial_out_lhs | 1/1 | 1/1 | True | 1/1 | False | False | False | None |
| double_implicit_contracted | lazy off | spatial_primal_rhs | 1/1 | 1/1 | True | 1/1 | False | False | False | None |
| double_implicit_contracted | lazy nodemote | batch_out | 2/2 | 2/2 | True | 2/2 | False | False | False | None |
| double_implicit_contracted | lazy nodemote | spatial_out_lhs | 1/1 | 1/1 | True | 1/1 | False | False | False | None |
| double_implicit_contracted | lazy nodemote | spatial_primal_rhs | 1/1 | 1/1 | True | 1/1 | False | False | False | None |
| double_implicit_contracted | lazy full | batch_out | 2/2 | 2/2 | True | 2/2 | False | False | False | None |
| double_implicit_contracted | lazy full | spatial_out_lhs | 1/1 | 1/1 | True | 1/1 | False | False | False | None |
| double_implicit_contracted | lazy full | spatial_primal_rhs | 1/1 | 1/1 | True | 1/1 | False | False | False | None |
| double_implicit_batch | lazy off | contract | 1/1 | 1/1 | True | 1/1 | False | False | False | None |
| double_implicit_batch | lazy off | batch_out | 2/2 | 2/2 | True | 1/1 | False | False | False | None |
| double_implicit_batch | lazy off | spatial_out_lhs | 1/1 | 1/1 | True | 1/1 | False | False | False | None |
| double_implicit_batch | lazy off | spatial_primal_rhs | 1/1 | 1/1 | True | 1/1 | False | False | False | None |
| double_implicit_batch | lazy nodemote | contract | 1/1 | 1/1 | True | 1/1 | False | False | False | None |
| double_implicit_batch | lazy nodemote | batch_out | 2/2 | 2/2 | True | 1/1 | True | False | False | None |
| double_implicit_batch | lazy nodemote | spatial_out_lhs | 1/1 | 1/1 | True | 1/1 | False | False | False | None |
| double_implicit_batch | lazy nodemote | spatial_primal_rhs | 1/1 | 1/1 | True | 1/1 | False | False | False | None |
| double_implicit_batch | lazy full | contract | 1/1 | 1/1 | True | 1/1 | False | False | False | None |
| double_implicit_batch | lazy full | batch_out | 2/2 | 2/2 | True | 1/1 | True | False | False | None |
| double_implicit_batch | lazy full | spatial_out_lhs | 1/1 | 1/1 | True | 1/1 | False | False | False | None |
| double_implicit_batch | lazy full | spatial_primal_rhs | 1/1 | 1/1 | True | 1/1 | False | False | False | None |
| uniform_operand | lazy off | contract | 1/1 | 1/1 | True | 1/1 | False | False | False | None |
| uniform_operand | lazy off | batch_out | 2/2 | 2/2 | True | 1/2 | False | False | False | None |
| uniform_operand | lazy off | spatial_out_lhs | 1/1 | 1/1 | True | 1/1 | False | False | False | None |
| uniform_operand | lazy off | spatial_primal_rhs | 1/1 | 1/1 | True | 1/1 | False | False | False | None |
| uniform_operand | lazy nodemote | contract | 1/1 | 1/1 | True | 1/1 | False | False | False | None |
| uniform_operand | lazy nodemote | batch_out | 2/2 | 2/2 | True | 1/2 | False | False | False | None |
| uniform_operand | lazy nodemote | spatial_out_lhs | 1/1 | 1/1 | True | 1/1 | False | True | False | None |
| uniform_operand | lazy nodemote | spatial_primal_rhs | 1/1 | 1/1 | True | 1/1 | False | False | False | None |
| uniform_operand | lazy full | contract | 1/1 | 1/1 | True | 1/1 | False | False | False | None |
| uniform_operand | lazy full | batch_out | 2/2 | 2/2 | True | 1/2 | False | False | False | l |
| uniform_operand | lazy full | spatial_out_lhs | 1/1 | 1/1 | True | 1/1 | False | True | False | None |
| uniform_operand | lazy full | spatial_primal_rhs | 1/1 | 1/1 | True | 1/1 | False | False | False | None |
| lcm_grid | lazy off | contract | 4/6 | 12/2 | False | 12/12 | False | False | False | None |
| lcm_grid | lazy nodemote | contract | 4/6 | 12/2 | False | 12/12 | False | False | False | None |
| lcm_grid | lazy full | contract | 4/6 | 12/2 | False | 12/12 | False | False | False | None |
| spatial_sparse | lazy off | contract | 1/1 | 1/1 | True | 1/1 | False | False | False | None |
| spatial_sparse | lazy off | spatial_out_lhs | 1/1 | 1/1 | True | 1/1 | False | False | False | None |
| spatial_sparse | lazy off | spatial_sparse_lhs | 3/1 | 3/1 | True | 3/1 | False | False | False | None |
| spatial_sparse | lazy off | spatial_primal_rhs | 1/1 | 1/1 | True | 1/1 | False | False | False | None |
| spatial_sparse | lazy nodemote | contract | 1/1 | 1/1 | True | 1/1 | False | False | False | None |
| spatial_sparse | lazy nodemote | spatial_out_lhs | 1/1 | 1/1 | True | 1/1 | False | False | False | None |
| spatial_sparse | lazy nodemote | spatial_sparse_lhs | 3/1 | 3/1 | True | 3/1 | False | False | False | None |
| spatial_sparse | lazy nodemote | spatial_primal_rhs | 1/1 | 1/1 | True | 1/1 | False | False | False | None |
| spatial_sparse | lazy full | contract | 1/1 | 1/1 | True | 1/1 | False | False | False | None |
| spatial_sparse | lazy full | spatial_out_lhs | 1/1 | 1/1 | True | 1/1 | False | False | False | None |
| spatial_sparse | lazy full | spatial_sparse_lhs | 3/1 | 3/1 | True | 3/1 | False | False | False | None |
| spatial_sparse | lazy full | spatial_primal_rhs | 1/1 | 1/1 | True | 1/1 | False | False | False | None |
| no_implicit_control | lazy off | contract | 4/4 | 4/4 | True | 4/4 | False | False | False | None |
| no_implicit_control | lazy nodemote | contract | 4/4 | 4/4 | True | 4/4 | False | False | False | None |
| no_implicit_control | lazy full | contract | 4/4 | 4/4 | True | 4/4 | False | False | False | None |

## Table E — values against the dense oracle

| case | mode | max abs err | oracle scale |
|---|---|---|---|
| single_implicit_sparse | incumbent | 0.0 | 3.376897096633911 |
| single_implicit_sparse | lazy off | 0.0 | 3.376897096633911 |
| single_implicit_sparse | lazy nodemote | 0.0 | 3.376897096633911 |
| single_implicit_sparse | lazy full | 1.1920928955078125e-07 | 3.376897096633911 |
| single_implicit_sparse | planner | 1.1920928955078125e-07 | 3.376897096633911 |
| single_implicit_sparse_rhs | incumbent | 0.0 | 3.044081687927246 |
| single_implicit_sparse_rhs | lazy off | 0.0 | 3.044081687927246 |
| single_implicit_sparse_rhs | lazy nodemote | 0.0 | 3.044081687927246 |
| single_implicit_sparse_rhs | lazy full | 0.0 | 3.044081687927246 |
| single_implicit_sparse_rhs | planner | 1.1920928955078125e-07 | 3.044081687927246 |
| single_implicit_dense_block | incumbent | 0.0 | 5.932111740112305 |
| single_implicit_dense_block | lazy off | 0.0 | 5.932111740112305 |
| single_implicit_dense_block | lazy nodemote | 0.0 | 5.932111740112305 |
| single_implicit_dense_block | lazy full | 0.0 | 5.932111740112305 |
| single_implicit_dense_block | planner | 0.0 | 5.932111740112305 |
| single_implicit_dense_contracted | incumbent | 0.0 | 4.9510297775268555 |
| single_implicit_dense_contracted | lazy off | 0.0 | 4.9510297775268555 |
| single_implicit_dense_contracted | lazy nodemote | 2.384185791015625e-07 | 4.9510297775268555 |
| single_implicit_dense_contracted | lazy full | 2.384185791015625e-07 | 4.9510297775268555 |
| single_implicit_dense_contracted | planner | 2.384185791015625e-07 | 4.9510297775268555 |
| single_implicit_dense_carried | incumbent | 0.0 | 5.720456123352051 |
| single_implicit_dense_carried | lazy off | 0.0 | 5.720456123352051 |
| single_implicit_dense_carried | lazy nodemote | 0.0 | 5.720456123352051 |
| single_implicit_dense_carried | lazy full | 2.384185791015625e-07 | 5.720456123352051 |
| single_implicit_dense_carried | planner | 2.384185791015625e-07 | 5.720456123352051 |
| double_implicit_contracted | incumbent | 2.384185791015625e-07 | 4.49523401260376 |
| double_implicit_contracted | lazy off | 2.384185791015625e-07 | 4.49523401260376 |
| double_implicit_contracted | lazy nodemote | 2.384185791015625e-07 | 4.49523401260376 |
| double_implicit_contracted | lazy full | 2.384185791015625e-07 | 4.49523401260376 |
| double_implicit_contracted | planner | 2.384185791015625e-07 | 4.49523401260376 |
| double_implicit_batch | incumbent | 0.0 | 4.442869663238525 |
| double_implicit_batch | lazy off | 0.0 | 4.442869663238525 |
| double_implicit_batch | lazy nodemote | 0.0 | 4.442869663238525 |
| double_implicit_batch | lazy full | 0.0 | 4.442869663238525 |
| double_implicit_batch | planner | 0.0 | 4.442869663238525 |
| uniform_operand | incumbent | 0.0 | 3.908705472946167 |
| uniform_operand | lazy off | 0.0 | 3.908705472946167 |
| uniform_operand | lazy nodemote | 0.0 | 3.908705472946167 |
| uniform_operand | lazy full | 0.0 | 3.908705472946167 |
| uniform_operand | planner | 0.0 | 3.908705472946167 |
| lcm_grid | incumbent | 4.76837158203125e-07 | 5.4027557373046875 |
| lcm_grid | lazy off | 4.76837158203125e-07 | 5.4027557373046875 |
| lcm_grid | lazy nodemote | 4.76837158203125e-07 | 5.4027557373046875 |
| lcm_grid | lazy full | 4.76837158203125e-07 | 5.4027557373046875 |
| lcm_grid | planner | 0.0 | 5.4027557373046875 |
| spatial_sparse | incumbent | 4.76837158203125e-07 | 6.244970321655273 |
| spatial_sparse | lazy off | 4.76837158203125e-07 | 6.244970321655273 |
| spatial_sparse | lazy nodemote | 4.76837158203125e-07 | 6.244970321655273 |
| spatial_sparse | lazy full | 4.76837158203125e-07 | 6.244970321655273 |
| spatial_sparse | planner | 0.0 | 6.244970321655273 |
| no_implicit_control | incumbent | 0.0 | 4.635643482208252 |
| no_implicit_control | lazy off | 0.0 | 4.635643482208252 |
| no_implicit_control | lazy nodemote | 0.0 | 4.635643482208252 |
| no_implicit_control | lazy full | 0.0 | 4.635643482208252 |
| no_implicit_control | planner | 0.0 | 4.635643482208252 |
