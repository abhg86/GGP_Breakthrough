package AIs;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ThreadLocalRandom;

import game.Game;
import main.collections.FastArrayList;
import other.AI;
import other.RankUtils;
import other.context.Context;
import other.move.Move;
import other.trial.Trial;



/**
 * A simple implementation of a UCT approach in imperfect information.
 * 
 * Only supports deterministic, alternating-move games.
 * 
 * @author Aymeric Behaegel
 */

public class HiddenUCT extends AI{	
	//-------------------------------------------------------------------------
	
	/** Our player index */
	protected int player = -1;

	/** The layer of the tree that comes after the current state of the game*/
	protected Set<Node> followingLayer = new HashSet<>();

	/** The root node of the tree */
	protected Node root = null;
	
	//-------------------------------------------------------------------------
	
	/**
	 * Constructor
	 */
	public HiddenUCT()
	{
		this.friendlyName = "Hidden UCT";
	}
	
	//-------------------------------------------------------------------------

	@Override
	public Move selectAction
	(
		final Game game,
		final Context context, 
		final double maxSeconds, 
		final int maxIterations, 
		final int maxDepth
	)
	{
		// Return the only available action if there is only one (for example for a pass)
		if (context.game().moves(context).moves().size() == 1)
			return context.game().moves(context).moves().get(0);

		// Start out by creating a new root node if it doesn't exist yet
		if (root == null){
			Trial trial = new Trial(context.game());
			Context startingContext = new Context(context.game(), trial);
			game.start(startingContext);
			root = new Node(null, null, startingContext);
		}
		
		// We'll respect any limitations on max seconds and max iterations (don't care about max depth)
		final long stopTime = (maxSeconds > 0.0) ? System.currentTimeMillis() + (long) (maxSeconds * 1000L) : Long.MAX_VALUE;
		final int maxIts = (maxIterations >= 0) ? maxIterations : Integer.MAX_VALUE;
		
		int numIterations = 0;

		// Moves played before the current state of the game
		List<Move> realMoves = context.trial().generateRealMovesList();
		System.out.println("Real moves : " + realMoves);
		List<Context> realContexts = new ArrayList<Context>();
		
		// Our main loop through MCTS iterations
		while 
		(
			numIterations < maxIts && 					// Respect iteration limit
			System.currentTimeMillis() < stopTime && 	// Respect time limit
			!wantsInterrupt								// Respect GUI user clicking the pause button
		)
		{
			// Start in root node
			Node current = root;
			if (realContexts.isEmpty()){
				realContexts.add(root.context);
			}
			Context realContext = new Context(root.context);
			
			int nbMoves = 0;
			// Traverse tree
			while (true)
			{
				if (current.context.trial().over())
				{
					// We've reached a terminal state
					break;
				}
				
				if (! current.possible){
					// We're in a node that is impossible
					break;
				}
				// System.out.println("nbMoves : " + nbMoves);

				if (nbMoves < realMoves.size()){
					if (nbMoves < realContexts.size() -1){
						realContext = realContexts.get(nbMoves);
					}
					else {
						realContext = new Context(realContext);
						realContext.game().apply(realContext, realMoves.get(nbMoves));
						realContexts.add(realContext);
					}

					if (current.context.state().mover() == player ){
						// We're in a node corresponding to a move of the player that has already been played
						current = select(current, realMoves.get(nbMoves), realContext);
					} else {
						// We're in a node corresponding to a move of the opponent but before the current state of the game
						current = select(current, null, realContext);
					}
				} else {
					// We're in a node corresponding to after the current state of the game
					current = select(current, null, null);

					if (nbMoves == realMoves.size()) { followingLayer.add(current);}
				}
				
				if (current.visitCount == 0)
				{
					// We've expanded a new node, time for playout!
					break;
				}
				nbMoves++;
			}
			
			Context contextEnd = current.context;
			
			if (!contextEnd.trial().over() && current.possible)
			{
				// Run a playout if we don't already have a terminal game state in node
				contextEnd = new Context(contextEnd);
				game.playout
				(
					contextEnd, 
					null, 
					-1.0, 
					null, 
					0, 
					-1, 
					ThreadLocalRandom.current()
				);
			}
			
			// This computes utilities for all players at the of the playout,
			// which will all be values in [-1.0, 1.0]
			final double[] utilities = RankUtils.utilities(contextEnd);
			
			// Backpropagate utilities through the tree
			while (current != null)
			{
				current.visitCount += 1;
				for (int p = 1; p <= game.players().count(); ++p)
				{
					current.scoreSums[p] += utilities[p];
				}
				current = current.parent;
			}
			
			// Increment iteration count
			++numIterations;
		}

		
		// Return the move we wish to play
		Move chosenMove = finalMoveSelection(context);
		// return chosenMove;

		final Context contextFinal = new Context(context);
		if (game.moves(contextFinal).moves().contains(chosenMove)){
			return chosenMove;
		}
		else {
			return game.moves(contextFinal).moves().get(0);
		}
	}
	
	/**
	 * Selects child of the given "current" node according to UCB1 equation.
	 * This method also implements the "Expansion" phase of MCTS, and creates
	 * a new node if the given current node has unexpanded moves.
	 * 
	 * @param current
	 * @return Selected node (if it has 0 visits, it will be a newly-expanded node).
	 */
	public Node select(final Node current, final Move realMove, Context realContext)
	{
		if (realMove != null){
			// We're in a node corresponding to a move of the player that has already been played so we expand only toward this move
			current.unexpandedMoves.clear();
			final Context context = new Context(current.context);
			try {
				context.game().apply(context, realMove);
				if (isCoherent(realContext, context)){
					Node newNode = new Node(current, realMove, context);
					newNode.visitCount = 1;
					return newNode;
				}
				else {
					Node impossibleNode = new Node(current, realMove, context);
					impossibleNode.visitCount = 1;
					impossibleNode.scoreSums[player] = -1.0;
					impossibleNode.unexpandedMoves.clear();
					impossibleNode.possible = false;
					propagateImpossible(current);
					return impossibleNode;
				}
			} catch (Exception e) {
				// The move is not legal here so we are in the wrong world 
				current.visitCount = 1;
				current.scoreSums[player] = -1.0;
				current.possible = false;
				propagateImpossible(current);
				return current;
			}
		}
		
		if (!current.unexpandedMoves.isEmpty())
		{
			// randomly select an unexpanded move
			final Move move = current.unexpandedMoves.remove(
					ThreadLocalRandom.current().nextInt(current.unexpandedMoves.size()));
			
			// create a copy of context
			final Context context = new Context(current.context);
			
			// apply the move
			context.game().apply(context, move);
			if (realContext != null){
				if ( isCoherent(context, realContext)){
					Node newNode = new Node(current, move, context);
					newNode.visitCount = 1;
					return newNode;
				} else {
					Node impossibleNode = new Node(current, move, context);
					impossibleNode.visitCount = 1;
					impossibleNode.scoreSums[player] = -1.0;
					impossibleNode.unexpandedMoves.clear();
					impossibleNode.possible = false;
					propagateImpossible(current);
					return impossibleNode;
				}
			}
			// create new node and return it
			return new Node(current, move, context);
		}
		
		// use UCB1 equation to select from all children, with random tie-breaking
		Node bestChild = null;
        double bestValue = Double.NEGATIVE_INFINITY;
        final double twoParentLog = 2.0 * Math.log(Math.max(1, current.visitCount));
        int numBestFound = 0;
        
        final int numChildren = current.children.size();
        final int mover = current.context.state().mover();

        for (int i = 0; i < numChildren; ++i) 
        {
        	final Node child = current.children.get(i);
        	final double exploit = child.scoreSums[mover] / child.visitCount;
        	final double explore = Math.sqrt(twoParentLog / child.visitCount);
        
            final double ucb1Value = exploit + explore;
            
            if (ucb1Value > bestValue)
            {
                bestValue = ucb1Value;
                bestChild = child;
                numBestFound = 1;
            }
            else if 
            (
            	ucb1Value == bestValue && 
            	ThreadLocalRandom.current().nextInt() % ++numBestFound == 0
            )
            {
            	// this case implements random tie-breaking
            	bestChild = child;
            }
        }
        
        return bestChild;
	}
	
	/**
	 * Selects the move we wish to play using the "Robust Child" strategy
	 * (meaning that we play the move leading to the nodes
	 * with the highest visit counts).
	 * 
	 * @param rootNode
	 * @return
	 */
	public Move finalMoveSelection(Context context)
	{
		Map<Move, Integer> moveVisits = new HashMap<>();
		for (Node node : followingLayer) {
			if (node.possible){
				if (moveVisits.containsKey(node.moveFromParent)){
					moveVisits.put(node.moveFromParent, moveVisits.get(node.moveFromParent) + node.visitCount);
				} else {
					moveVisits.put(node.moveFromParent, node.visitCount);
				}
			}
		}
		return moveVisits.entrySet().stream().max(Map.Entry.comparingByValue()).get().getKey();
		// while (moveVisits.size() > 0){
			// if (context.game().moves(context).moves().contains(move)){
			// 	System.out.println("Move found");
			// 	return move;
			// } else {
			// 	moveVisits.remove(move);
			// }
		// }
		// throw new RuntimeException("No move found");
	}

	private boolean isCoherent(Context context, Context predictedContext){
		if (context.state().owned().sites(player).equals(predictedContext.state().owned().sites(player))){
			return true;
		}
		else {
			// if (context.trial().numMoves() > 3){
			// 	System.out.println("Coherent : " + context.state().owned().sites(player));
			// 	System.out.println("Predicted : " + predictedContext.state().owned().sites(player));
			// }
			return false;
		}
	}

	private void propagateImpossible(Node parent){
		if (!parent.unexpandedMoves.isEmpty()){
			return;
		}
		boolean allChildImpossible = true;
		for (Node child : parent.children){
			if (child.possible){
				allChildImpossible = false;
			}
		}
		if (allChildImpossible){
			parent.possible = false;
			if (parent.parent != null) {
				propagateImpossible(parent.parent);
			}
		}
	}
	
	@Override
	public void initAI(final Game game, final int playerID)
	{
		this.player = playerID;

	}
	
	@Override
	public boolean supportsGame(final Game game)
	{
		if (game.isStochasticGame())
			return false;
		
		if (!game.isAlternatingMoveGame())
			return false;
		
		return true;
	}
	
	//-------------------------------------------------------------------------
	
	/**
	 * Inner class for nodes used by Hidden UCT
	 * 
	 * @author Aymeric Behaegel
	 */
	private static class Node
	{
		/** Our parent node */
		private final Node parent;
		
		/** The move that led from parent to this node */
		private final Move moveFromParent;
		
		/** This objects contains the game state for this node (this is why we don't support stochastic games) */
		private final Context context;
		
		/** Visit count for this node */
		private int visitCount = 0;

		/** Depth of the node in the tree */
		private int depth = 0;

		/** If the node is possible or not */
		private boolean possible = true;
		
		/** For every player, sum of utilities / scores backpropagated through this node */
		private final double[] scoreSums;
		
		/** Child nodes */
		private final List<Node> children = new ArrayList<Node>();
		
		/** List of moves for which we did not yet create a child node */
		private final FastArrayList<Move> unexpandedMoves;
		
		/**
		 * Constructor
		 * 
		 * @param parent
		 * @param moveFromParent
		 * @param context
		 */
		public Node(final Node parent, final Move moveFromParent, final Context context)
		{
			this.parent = parent;
			this.moveFromParent = moveFromParent;
			if (parent != null){
				depth = parent.depth + 1;
			}
			this.context = context;
			final Game game = context.game();
			scoreSums = new double[game.players().count() + 1];
			
			// For simplicity, we just take ALL legal moves. 
			// This means we do not support simultaneous-move games.
			unexpandedMoves = new FastArrayList<Move>(game.moves(context).moves());
			
			if (parent != null)
				parent.children.add(this);
		}
		
	}
	
	//-------------------------------------------------------------------------

}
