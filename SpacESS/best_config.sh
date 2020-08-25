# Best Configuration for RotatE
#
bash run.sh train RotatE FB15k 0 0 1024 256 1000 24.0 1.0 0.0001 150000 16 -de
bash run.sh train RotatE FB15k-237 0 0 1024 256 1000 9.0 1.0 0.00005 100000 16 -de
bash run.sh train RotatE wn18 0 0 512 1024 500 12.0 0.5 0.0001 80000 8 -de
bash run.sh train RotatE wn18rr 0 0 512 1024 500 6.0 0.5 0.00005 80000 8 -de
bash run.sh train RotatE countries_S1 0 0 512 64 1000 0.1 1.0 0.000002 40000 8 -de --countries
bash run.sh train RotatE countries_S2 0 0 512 64 1000 0.1 1.0 0.000002 40000 8 -de --countries 
bash run.sh train RotatE countries_S3 0 0 512 64 1000 0.1 1.0 0.000002 40000 8 -de --countries
bash run.sh train RotatE YAGO3-10 0 0 1024 400 500 24.0 1.0 0.0002 100000 4 -de
#
# Best Configuration for pRotatE
#
bash run.sh train pRotatE FB15k 0 0 1024 256 1000 24.0 1.0 0.0001 150000 16
bash run.sh train pRotatE FB15k-237 0 0 1024 256 1000 9.0 1.0 0.00005 100000 16
bash run.sh train pRotatE wn18 0 0 512 1024 500 12.0 0.5 0.0001 80000 8
bash run.sh train pRotatE wn18rr 0 0 512 1024 500 6.0 0.5 0.00005 80000 8
bash run.sh train pRotatE countries_S1 0 0 512 64 1000 0.1 1.0 0.000002 40000 8 --countries
bash run.sh train pRotatE countries_S2 0 0 512 64 1000 0.1 1.0 0.000002 40000 8 --countries
bash run.sh train pRotatE countries_S3 0 0 512 64 1000 0.1 1.0 0.000002 40000 8 --countries
#
# Best Configuration for TransE
# 
bash run.sh train TransE FB15k 0 0 1024 256 1000 24.0 1.0 0.0001 150000 16
bash run.sh train TransE FB15k-237 0 0 1024 256 1000 9.0 1.0 0.00005 100000 16
bash run.sh train TransE wn18 0 0 512 1024 500 12.0 0.5 0.0001 80000 8
bash run.sh train TransE wn18rr 0 0 512 1024 500 6.0 0.5 0.00005 80000 8
bash run.sh train TransE countries_S1 0 0 512 64 1000 0.1 1.0 0.000002 40000 8 --countries
bash run.sh train TransE countries_S2 0 0 512 64 1000 0.1 1.0 0.000002 40000 8 --countries
bash run.sh train TransE countries_S3 0 0 512 64 1000 0.1 1.0 0.000002 40000 8 --countries
#
# Best Configuration for ComplEx
# 
bash run.sh train ComplEx FB15k 0 0 1024 256 1000 500.0 1.0 0.001 150000 16 -de -dr -r 0.000002
bash run.sh train ComplEx FB15k-237 0 0 1024 256 1000 200.0 1.0 0.001 100000 16 -de -dr -r 0.00001
bash run.sh train ComplEx wn18 0 0 512 1024 500 200.0 1.0 0.001 80000 8 -de -dr -r 0.00001
bash run.sh train ComplEx wn18rr 0 0 512 1024 500 200.0 1.0 0.002 80000 8 -de -dr -r 0.000005
bash run.sh train ComplEx countries_S1 0 0 512 64 1000 1.0 1.0 0.000002 40000 8 -de -dr -r 0.0005 --countries
bash run.sh train ComplEx countries_S2 0 0 512 64 1000 1.0 1.0 0.000002 40000 8 -de -dr -r 0.0005 --countries
bash run.sh train ComplEx countries_S3 0 0 512 64 1000 1.0 1.0 0.000002 40000 8 -de -dr -r 0.0005 --countries
#
# Best Configuration for DistMult
# 
bash run.sh train DistMult FB15k 0 relu_logsimoid 1024 256 1000 500.0 1.0 0.001 150000 16 -de -dr -r 0.000002
bash run.sh train DistMult FB15k-237 0 0 1024 256 1000 200.0 1.0 0.001 100000 16 -de -dr -r 0.00001
bash run.sh train DistMult wn18 0 0 512 1024 500 200.0 1.0 0.001 80000 8 -de -dr -r 0.00001
bash run.sh train DistMult wn18rr 0 0 512 1024 500 200.0 1.0 0.002 80000 8 -de -dr -r 0.000005
bash run.sh train DistMult countries_S1 0 0 512 64 1000 1.0 1.0 0.000002 40000 8 -de -dr -r 0.0005 --countries
bash run.sh train DistMult countries_S2 0 0 512 64 1000 1.0 1.0 0.000002 40000 8 -de -dr -r 0.0005 --countries
bash run.sh train DistMult countries_S3 0 0 512 64 1000 1.0 1.0 0.000002 40000 8 -de -dr -r 0.0005 --countries
#




#
# Configs for gamma hyperparams (gamma1 & gamma2) for experiment
#
bash run.sh experiment TransE FB15k 3 relu 1024 10 200 24.0 1.0 0.0001 150000 16 .3
bash run.sh experiment TransE FB15k 1 04 1024 10 100 24.0 1.0 0.0001 150000 16 .4
bash run.sh experiment TransE FB15k 1 07 1024 10 100 24.0 1.0 0.0001 150000 16 .7
bash run.sh experiment TransE FB15k 1 10 1024 10 100 24.0 1.0 0.0001 150000 16 1
bash run.sh experiment TransE FB15k 0 0 1024 10 100 24.0 1.0 0.0001 150000 16 1.3
bash run.sh experiment TransE FB15k 0 0 1024 10 100 24.0 1.0 0.0001 150000 16 1.6
bash run.sh experiment TransE FB15k 0 0 1024 10 100 24.0 1.0 0.0001 150000 16 1.9
bash run.sh experiment RotatE wn18 0 0 512 1024 500 12.0 0.5 0.0001 80000 8 -de .3


# relu version parallelized
bash run.sh experiment TransE FB15k 1 01 1024 10 100 24.0 1.0 0.0001 150000 16 .3 .1 adam
bash run.sh experiment RotatE wn18rr 0 0 512 1024 500 6.0 0.5 0.00005 80000 8 -de .3

bash run.sh experiment RotatE wn18rr 0 0 512 10 200 12.0 0.5 0.0001 80000 8 .3 .1 adam -de
bash run.sh grid TransE wn18 0 0 512 10 200 12.0 0.5 0.0001 80000 8 .3 .1 adam


# new grid settings
# mode model dataset gpu save_id batch_size dim gamma alpha lr test_batch_size
bash run.sh grid TransRotatE FB15k 1 0 1024 200 24.0 1.0 0.0001 16 -de > transrotate_fb15k_adam.csv
bash run.sh train TransE FB15k 2 0 1024 50 300 24.0 1.0 0.0001 150000 16 .6 .1 adam

bash run.sh train TransE FB15k 0 adaptive 1024 100 24.0 1.0 0.0001 150000 16 --loss adaptive_margin > adaptive



# rules settings
bash run.sh grid TransE FB15k 3 adapt_adam2 1024 100 24.0 1.0 0.0001 16 > transe_fb15k_adaptive2.csv
bash run.sh grid RotatE FB15k 2 rotate_loss 1024 300 24.0 1.0 0.0001 16 -de --loss rotate > fb_rotate_loss.csv
bash run.sh grid RotatE wn18 3 wn_adapt 512 200 12.0 0.5 0.0001 8 -de > rotate_wn18_adaptive.csv


bash run.sh grid TransRotatE FB15k 2 implication 1024 500 24.0 .5 0.1 16 -de --impl > transrot_implication.csv
bash run.sh grid TransRotatE FB15k 1 transrot_inverse 1024 500 24.0 .5 0.1 16 --loss adaptive_margin --inv -de > transrotate_fb_inverse.csv
bash run.sh grid TransRotatE FB15K 0 ada_transrot 1024 300 24.0 0.5 0.1 16 -dr -de --loss adaptive_margin > transrot_updated_ada.csv


bash run.sh grid TransRotatE FB15k 2 norule 1024 500 24.0 .5 0.1 16 -de > transrot_norule.csv
bash run.sh grid TransRotatE FB15k 1 implication_smallbs 1024 500 24.0 .5 0.1 16 -de --impl > transrot_impl_small_batch.csv
bash run.sh grid TransRotatE FB15k 1 inverse_short 1024 500 24.0 .5 0.1 16 -de --inv > transrot_inverse_short.csv
bash run.sh grid TransRotatE FB15k 2 inverse_eps1 1024 500 24.0 .5 0.1 16 -de --inv > transrot_inverse_eps1.csv
