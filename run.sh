#        seed_val                      dataset   fitα    lidx     split_ratios   mixing_parameter
# julia main.jl  0         county_election_2012  true     1:1    0.05:0.05:0.60 0.999 2>&1 >> county_income &
# julia main.jl  0         county_election_2012  true     5:5    0.05:0.05:0.60 0.999 2>&1 >> county_education &
# julia main.jl  0         county_election_2012  true     6:6    0.05:0.05:0.60 0.999 2>&1 >> county_unemployment &
# julia main.jl  0         county_election_2012  true     7:7    0.05:0.05:0.60 0.999 2>&1 >> county_election &
# julia main.jl  0       environment_pm2.5_2008  true     1:1    0.05:0.05:0.60       2>&1 >> environment_airT &
# julia main.jl  0       environment_pm2.5_2008  true     2:2    0.05:0.05:0.60       2>&1 >> environment_landT &
# julia main.jl  0       environment_pm2.5_2008  true     3:3    0.05:0.05:0.60       2>&1 >> environment_precipitation &
# julia main.jl  0       environment_pm2.5_2008  true     4:4    0.05:0.05:0.60       2>&1 >> environment_sunlight &
# julia main.jl  0       environment_pm2.5_2008  true     5:5    0.05:0.05:0.60       2>&1 >> environment_pm2.5 &
# julia main.jl  0           ward_election_2012  true     1:1    0.05:0.05:0.60 0.999 2>&1 >> ward_edu &
# julia main.jl  0           ward_election_2012  true     2:2    0.05:0.05:0.60 0.999 2>&1 >> ward_age &
# julia main.jl  0           ward_election_2012  true     3:3    0.05:0.05:0.60 0.999 2>&1 >> ward_gender &
# julia main.jl  0           ward_election_2012  true     4:4    0.05:0.05:0.60 0.999 2>&1 >> ward_income &
# julia main.jl  0           ward_election_2012  true     5:5    0.05:0.05:0.60 0.999 2>&1 >> ward_populationsize &
# julia main.jl  0           ward_election_2012  true     6:6    0.05:0.05:0.60 0.999 2>&1 >> ward_election &
# julia main.jl  0          twitch_PTBR_true_04  true     5:5    0.05:0.05:0.60 0.900 2>&1 >> twitch_PTBR_04 &
# julia main.jl  0          twitch_PTBR_true_08  true     9:9    0.05:0.05:0.60 0.900 2>&1 >> twitch_PTBR_08 &
# julia main.jl  0          twitch_PTBR_true_16  true    17:17   0.05:0.05:0.60       2>&1 >> twitch_PTBR_16 &
# julia main.jl  0          twitch_PTBR_true_32  true    33:33   0.05:0.05:0.60       2>&1 >> twitch_PTBR_32 &
# julia main.jl  0          twitch_PTBR_true_64  true    65:65   0.05:0.05:0.60       2>&1 >> twitch_PTBR_64 &
# julia main.jl  0                 cora_true_04  false    5:11   0.05:0.05:0.60 0.900 2>&1 >> cora_04 &
# julia main.jl  0                 cora_true_08  false    9:15   0.05:0.05:0.60 0.900 2>&1 >> cora_08 &
# julia main.jl  0                 cora_true_16  false   17:23   0.05:0.05:0.60 0.900 2>&1 >> cora_16 &
# julia main.jl  0                 cora_true_32  false   33:39   0.05:0.05:0.60 0.999 2>&1 >> cora_32 &
# julia main.jl  0                 cora_true_64  false   65:71   0.05:0.05:0.60 0.999 2>&1 >> cora_64 &
# julia main.jl  0                cora_false_00  false 1434:1440 0.05:0.05:0.60 0.900 2>&1 >> cora_ff &
  julia main.jl  0                       Amazon  false   26:27   0.05:0.05:0.60 0.900 2>&1 >> amazon  &
# julia main.jl  0  cropsim_harvestarea_2000_04  false   05:05   0.05:0.05:0.60 0.999 2>&1 >> hv00_05 &
# julia main.jl  0  cropsim_harvestarea_2000_09  false   10:10   0.05:0.05:0.60 0.999 2>&1 >> hv00_10 &
# julia main.jl  0  cropsim_harvestarea_2000_14  false   15:15   0.05:0.05:0.60 0.999 2>&1 >> hv00_15 &
# julia main.jl  0  cropsim_harvestarea_2000_19  false   20:20   0.05:0.05:0.60 0.999 2>&1 >> hv00_20 &
