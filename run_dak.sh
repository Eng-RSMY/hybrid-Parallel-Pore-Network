dakota -i 1_d_std.in >log.out
sed '1d' f.dat > f-mod.dat
python stat.py
