#= ALIASES =#

alias mlpy="if [ ! -f ~/Desktop/mlpy ]; then
		rm -fr ~/Desktop/mlpy
	fi && \
	cp -r /home/isetbz/mlpy/ ~/Desktop/ && \
	bash /home/isetbz/mlpy.sh"

alias orange="conda activate orange3 && \
	python -m Orange.canvas"
