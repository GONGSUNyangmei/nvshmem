#!/bin/bash
set -u

OUT_DIR="${OUT_DIR:-./put_bw_results}"
mkdir -p "$OUT_DIR"

TS=$(date +%Y%m%d_%H%M%S)
CSV="$OUT_DIR/put_bw_${TS}.csv"
RAW_DIR="$OUT_DIR/raw_${TS}"
mkdir -p "$RAW_DIR"

echo "ctas,threads,scope,size_bytes,bw_GBps" > "$CSV"

for j in block warp thread
do
    for i in 1 2 4 8 16 32 64 128 256 512 1024 
    do
        TAG="c${i}_t32_s${j}"
        RAW="$RAW_DIR/${TAG}.txt"

        echo "===== Running: ctas=$i threads=32 scope=$j ====="

        mpirun \
            -x LD_LIBRARY_PATH \
            -x NVSHMEM_BOOTSTRAP=MPI \
            -x NVSHMEM_DISABLE_GDRCOPY=true \
            -x NVSHMEM_IB_ENABLE_IBGDA=true \
            -x NVSHMEM_IBGDA_NUM_RC_PER_PE=$i \
            -np 2 -host r4,r6 \
            ./shmem_put_bw -c "$i" -t 32 -n 10000 -s "$j" -e 4194304\
            > "$RAW" 2>&1

        # 提取最终的测试数据表：从 "size(B)" 表头之后的纯数据行
        # 数据行格式示例: "4           None      0.001691"
        awk -v ctas="$i" -v threads=32 -v scope="$j" '
            /^size\(B\)/ { in_table=1; next }
            in_table {
                # 跳过空行和非数据行
                if ($0 ~ /^[[:space:]]*$/) next
                if ($1 ~ /^[0-9]+$/ && NF >= 3) {
                    printf "%s,%s,%s,%s,%s\n", ctas, threads, scope, $1, $NF
                }
            }
        ' "$RAW" | tee -a "$CSV"
    done
done

echo
echo "All results saved to: $CSV"
echo "Per-run raw logs in : $RAW_DIR"
