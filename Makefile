.DEFAULT_GOAL := exec
.PHONY :install exec test compile

NUM_RUN = 10

install:
	mvn install:install-file -Dfile=resources/Ludii-1.3.11.jar -DgroupId=org.Ludii -DartifactId=Ludii -Dversion=1.3.11 -Dpackaging=jar -DgeneratePom=true

compile:
	mvn clean compile

normal :
	mvn exec:java -Dexec.arguments="$(NUM_RUN),normal"

small_normal :
	mvn exec:java -Dexec.arguments="$(NUM_RUN),small_normal"

forced :
	mvn exec:java -Dexec.arguments="$(NUM_RUN),forced"

misere :
	mvn exec:java -Dexec.arguments="$(NUM_RUN),misere"

simultaneous :
	mvn exec:java -Dexec.arguments="$(NUM_RUN),simultaneous"

hidden :
	mvn exec:java -Dexec.arguments="$(NUM_RUN),hidden"

small_hidden :
	mvn exec:java -Dexec.arguments="$(NUM_RUN),small_hidden"