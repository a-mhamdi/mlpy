make:
	rsync -avr ~/MEGA/git-repos/isetbz/"Machine Learning"/Codes/ ./codes/
	rsync -avr ~/MEGA/git-repos/cosnip/Python/ml/ ./codes/
	# docker build -t pyml:latest .
