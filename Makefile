make:
	rsync -avr ~/MEGA/git-repos/isetbz/"Machine Learning"/Codes/ ./sl/
	rsync -avr ~/MEGA/git-repos/cosnip/Python/ml/ ./lab/
	# docker build -t pyml:latest .
