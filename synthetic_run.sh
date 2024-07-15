cudagpu=1
n_train=10000
ktrain=50

for trail in 1 2 3 4 5; do
	for bias in 0 1 2 3 4; do
		CUDA_VISIBLE_DEVICES=$cudagpu python synthetic.py --num_train ${n_train} --loadpath './result/synthetic/ntrain_'"$n_train"'_bias'"$bias"'_trail'"$trail"'/' --Ktrain ${ktrain} --regenerate True --bias ${bias} --style 'otm'		
		CUDA_VISIBLE_DEVICES=$cudagpu python synthetic.py --num_train ${n_train} --loadpath './result/synthetic/ntrain_'"$n_train"'_bias'"$bias"'_trail'"$trail"'/' --Ktrain ${ktrain} --regenerate False --bias ${bias} --style 'otm-bs'	
		CUDA_VISIBLE_DEVICES=$cudagpu python synthetic.py --num_train ${n_train} --loadpath './result/synthetic/ntrain_'"$n_train"'_bias'"$bias"'_trail'"$trail"'/' --Ktrain ${ktrain} --regenerate False --bias ${bias} --style 'otm-optest' 
		CUDA_VISIBLE_DEVICES=$cudagpu python synthetic.py --num_train ${n_train} --loadpath './result/synthetic/ntrain_'"$n_train"'_bias'"$bias"'_trail'"$trail"'/' --Ktrain ${ktrain} --regenerate False --bias ${bias} --style 'plt'
		CUDA_VISIBLE_DEVICES=$cudagpu python synthetic.py --num_train ${n_train} --loadpath './result/synthetic/ntrain_'"$n_train"'_bias'"$bias"'_trail'"$trail"'/' --Ktrain ${ktrain} --regenerate False --bias ${bias} --style 'tdm'
	done
done
