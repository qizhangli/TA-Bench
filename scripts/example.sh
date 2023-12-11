
method=LinBP_NewBackend
method=$(echo "$method" | tr '[:upper:]' '[:lower:]')
savedir="adv_img"
smodel=convnext_base

python3 -u attack.py --config="configs/${method}.py" \
--config.save_dir ${savedir} --config.model_name ${smodel}
python3 test.py --dir ${savedir}
