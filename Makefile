make:
	rsync -avr ~/MEGA/git-repos/isetbz/"Artificial Intelligence"/Codes/Python/ ./codes/
	rsync -avr ~/MEGA/git-repos/cosnip/Python/ml/ ./codes/
	# docker build -t ml-iset:latest .
