.DEFAULT_GOAL := exec
.PHONY :install exec test compile

NUM_RUN = 100

install:
	mvn install:install-file -Dfile=resources/Ludii-1.3.11.jar -DgroupId=org.Ludii -DartifactId=Ludii -Dversion=1.3.11 -Dpackaging=jar -DgeneratePom=true

compile:
	mvn clean compile

run :
	mvn exec:java -Dexec.arguments="$(NUM_RUN),forced"