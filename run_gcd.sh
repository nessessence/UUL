IMG_DIR="data_root/generated/study/esd-x-kv.bG.fG.T750-1000_0.10AhE0.40I0.80-N1.00G1.00_U.obama_sd1.4.bf16.bs4_r0/step1000/a photo of Barack Obama/7.50"
SAVE_EXCEL_PATH="data_root/score_logs/gcd"

python ./metrics/evaluate_by_gcd.py \
    --image_folder "${IMG_DIR}" \
    --save_excel_path "${SAVE_EXCEL_PATH}"