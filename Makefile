.DEFAULT_GOAL := exec
.PHONY :install exec test compile

NUM_RUN = 1

install:
	mvn install:install-file -Dfile=resources/Ludii-1.3.11.jar -DgroupId=org.Ludii -DartifactId=Ludii -Dversion=1.3.11 -Dpackaging=jar -DgeneratePom=true

compile:
	mvn clean compile

exec:
	parallel mvn exec:java -Dexec.arguments="{1},$(NUM_RUN)" ::: $(shell seq 1 1272)

test:
	parallel mvn exec:java -Dexec.arguments="{1},$(NUM_RUN)" ::: $(shell seq 7 8)

run :
	mvn exec:java -Dexec.arguments="$(NUM_RUN)"