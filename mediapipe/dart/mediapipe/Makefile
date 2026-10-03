ffigen:
	dart run ffigen --config ffigen.yaml

compile:
	gcc c/text_classifier.c -o c/text_classifier
	cd c && gcc -static -c -fPIC *.c -o text_classifier.o
	cd c && gcc -shared -o text_classifier.dylib text_classifier.o 

run:
	cd c && dart text_classifier_c.dart