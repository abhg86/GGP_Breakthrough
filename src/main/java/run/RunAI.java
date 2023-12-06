package run;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import game.Game;
import AIs.ExampleUCT;
import AIs.HiddenUCT;
import AIs.NormalUCD;
import AIs.RandomAI;
import AIs.UCD;
import other.AI;
import other.GameLoader;
import other.context.Context;
import other.model.Model;
import other.trial.Trial;
import utils.LudiiAI;
import search.mcts.MCTS;



public class RunAI
{
	public static void main(final String[] args)
	{
		int winrate = 0;
		for (int i=0; i<Integer.valueOf(args[0]); i++)
		{
			// Print memory available
			// Runtime runtime = Runtime.getRuntime();
			// long maxMemory = (runtime.maxMemory() / 1000000);
			// long allocatedMemory = (runtime.totalMemory() / 1000000);
			// long freeMemory = (runtime.freeMemory() / 1000000);
			// long usedMemory = allocatedMemory - freeMemory;
			// long availableMemory = maxMemory - usedMemory;
			// System.out.println("Max memory: " + maxMemory + " MB");
			// System.out.println("Available memory: " + availableMemory + " MB");
			// System.out.println("Allocated memory: " + allocatedMemory + " MB");
			// System.out.println("Free allocated memory: " + freeMemory + " MB");

			Game game;
			if (args[1].equals("normal")){
				game = GameLoader.loadGameFromName("Breakthrough.lud");
			} else if (args[1].equals("small_normal")){
				List<String> options = Arrays.asList("Board Size/4x4");
				game = GameLoader.loadGameFromName("Breakthrough.lud", options);
			} else if (args[1].equals("misere")){
				game = GameLoader.loadGameFromFile(new File("games/Breakthrough_misere.lud"));
			} else if (args[1].equals("forced")){
				game = GameLoader.loadGameFromFile(new File("games/Breakthrough_forced.lud"));
			} else if (args[1].equals("simultaneous")){
				game = GameLoader.loadGameFromFile(new File("games/Breakthrough_simultaneous.lud"));
			} else if (args[1].equals("hidden")){
				game = GameLoader.loadGameFromFile(new File("games/Breakthrough_hidden.lud"));
			} else if (args[1].equals("small_hidden")){
				List<String> options = Arrays.asList("Board Size/4x4");
				game = GameLoader.loadGameFromFile(new File("games/Breakthrough_hidden.lud"), options);
			}  else {
				game = GameLoader.loadGameFromName("Breakthrough.lud");
				System.out.println("Invalid game type, defaulting to normal");
			}
			
			Trial trial = new Trial(game);
			Context context = new Context(game, trial);
			game.start(context);
			
			System.out.println("We're playing " + game.name() + "!");
			
			// Create and init two agents
			final List<AI> ais = new ArrayList<AI>(3);
			ais.add(null);
			// ais.add(MCTS.createBiasedMCTS(0.3));
			// ais.add(new UCD());
			ais.add(new NormalUCD());
			// ais.add(new HiddenUCT());
			// ais.add(new ExampleUCT());
			ais.get(1).initAI(game, 1);

			// ais.add(new RandomAI());
			ais.add(new HiddenUCT());
			ais.get(2).initAI(game, 2);
			
			// This model object is the thing that will handle control flow for us
			final Model model = context.model();
			
			// Keep going until the game is over...
			while (!trial.over())
			{
				// The following call tells the model it should start a new "step"
				// using the given list of AIs, with 0.2 seconds of thinking time
				// per decision, per agent.
				//
				// The behaviour of this call depends on whether the model is for
				// alternating-move games or simultaneous-move games, but the basic
				// premise is that whatever AI(s) is (are) supposed to make a move
				// will start thinking about its move, and apply it to the game
				// state once the decision has been made.
				//
				// In an alternating-move game, this means a single agent is
				// queried to return a move, and that move is applied. In a
				// simultaneous-move game, it means that ALL active players are
				// queried to return moves, and they are all applied together.
				model.startNewStep(context, ais, 0.2);
				
				// In the following loop, we wait around until the model tells
				// us that it's ready with the processing of the step we asked
				// it to start.
				//
				// Generally this wouldn't be needed, because the startNewStep()
				// call should block and only return once any relevant AIs have
				// selected and applied their move to the game state. However,
				// the behaviour of the startNewStep() can be modified in various
				// ways by adding additional arguments. One possible modification
				// is to have it return immediately (which implies that any
				// AI-thinking must be performed in a separate Thread). A loop
				// like the one below may then, for example, be used for 
				// visualisation of the AI's thinking process, or for processing
				// human-player input in case any of the AIs were set to null.
				while (!model.isReady())
				{
					try
					{
						Thread.sleep(100);
					}
					catch (final InterruptedException e)
					{
						e.printStackTrace();
					}
				}
				
				// There is no need to explicitly apply any moves here anymore;
				// if model.isReady() returns true, this means we're ready for
				// the next time step!
			}
			

			// let's see what the result is
			System.out.println(context.trial().status());
			if (context.trial().status().winner() == 1)
				winrate++;
			System.out.println("Winrate: " + winrate + "/" + (i+1));
		}
		System.out.println("Winrate: " + winrate + "/" + args[0]);
	}
}


