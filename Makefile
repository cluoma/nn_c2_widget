all:
	gcc -o nn2 -O3 nn_c2.c -lm -lgsl -lopenblas -lpthread
