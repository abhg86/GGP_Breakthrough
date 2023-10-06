# GGP_Breakthrough


## How to launch it
First run 
```bash 
make install
``` 
to install Ludii-1.3.11.jar in your .m2 repository. 

(I don't know if necessary but) you can compile the java code with 
```bash
make compile
```

Then just run 
```bash
make normal
``` 
To launch 100 games of MCTS against Random in normal Breakthrough. You can replace "normal" by "misere", "forced" or "simultaneous" to have this modes of Breakthrough applied. 

Note that you will not see the graphic interface of Ludii by doing so, only results. If you want to use Ludii with a graphic interface you will have to go to the resources folder and launch the Ludii .jar file using the  ``` java -jar ``` command. Once on the Ludii window you can load a game from Ludii or from a .lud file (in folder games here) in the "File" option, or change the AI by clicking on the name of the players.