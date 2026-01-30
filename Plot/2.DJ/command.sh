# 1) Draw tke comparision

python draw_tke.py

# 2) Draw vorticity plots
python create_colorbar.py --vmax 100

python draw_vorticity.py --path DNS_data/velocity_10_200000.0_0.9960937500000927.npy --save DNS_1.0.pdf --vmax 100
python draw_vorticity.py --path DNS_data/velocity_10_200000.0_1.503906250000208.npy --save DNS_1.5.pdf --vmax 100
python draw_vorticity.py --path DNS_data/velocity_10_200000.0_1.9726562500003146.npy --save DNS_2.pdf --vmax 100


python draw_vorticity.py --path RTTD_data/u_time_0.97657.npy --v_path RTTD_data/v_time_0.97657.npy --save RTTD_1.0.pdf --vmax 100 --mps
python draw_vorticity.py --path RTTD_data/u_time_1.50391.npy --v_path RTTD_data/v_time_1.50391.npy --save RTTD_1.5.pdf --vmax 100 --mps
python draw_vorticity.py --path RTTD_data/u_time_1.99219.npy --v_path RTTD_data/v_time_1.99219.npy --save RTTD_2.0.pdf --vmax 100 --mps