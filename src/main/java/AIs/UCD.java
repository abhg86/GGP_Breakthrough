package AIs;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.Stack;
import java.util.concurrent.ThreadLocalRandom;


import utils.Pair;
import game.Game;
import gnu.trove.list.array.TIntArrayList;
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

public class UCD extends AI{	
	//-------------------------------------------------------------------------
	
	/** Our player index */
	protected int player = -1;

	/** The root node of the tree */
	protected Node root = null;
	protected Context startingContext = null;

	/** The maximum depth of the tree */
	protected int maxDepthReached = 0;

    /** The transposition table of existing nodes regrouped by similar observations from a player. 
     * Use symbolic edges to represent the data of all edges combined. 
	 * Have to use Arraylist for the key because pairs give different hashcode for the same content 
     */
	protected HashMap<ArrayList<Set<Integer>>, Node> transpoTable = new HashMap<ArrayList<Set<Integer>>, Node>();

	int d1 = 2;
	int d2 = 1;
	int d3 = 1;
	//-------------------------------------------------------------------------
	
	/**
	 * Constructor
	 */
	public UCD()
	{
		this.friendlyName = "UCD";
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
		// Start out by creating a new root node if it doesn't exist yet
		if (root == null){
			Trial trial = new Trial(context.game());
			startingContext = new Context(context.game(), trial);
			game.start(startingContext);
			
			//Compute id of the root context
			ArrayList<Set<Integer>> id = createID(startingContext);
			
			root = new Node(null, startingContext, id);
		}
		
		// We'll respect any limitations on max seconds and max iterations (don't care about max depth)
		final long stopTime = (maxSeconds > 0.0) ? System.currentTimeMillis() + (long) (maxSeconds * 1000L) : Long.MAX_VALUE;
		final int maxIts = (maxIterations >= 0) ? maxIterations : Integer.MAX_VALUE;
		
		int numIterations = 0;

		// Moves played before the current state of the game
		List<Move> realMoves = context.trial().generateRealMovesList();
		
		List<Context> realContexts = new ArrayList<Context>();
		List<ArrayList<Set<Integer>>> realIds = new ArrayList<ArrayList<Set<Integer>>>();
		
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
			Context realContext = new Context(startingContext);
			ArrayList<Set<Integer>> realId = createID(startingContext);
			if (realContexts.isEmpty()){
				realContexts.add(realContext);
				realIds.add(realId);
			}
			
			int nbMoves = 0;
			Stack<Edge> path = new Stack<Edge>();

			// Traverse tree
			while (true)
			{
				if (nbMoves < (realMoves.size() + 1)/2){
					// We're in a node corresponding to a move of the player that has already been played
					if (nbMoves < realContexts.size() -1){
						// We get the situation from the equivalent time for the real game if already computed
						realContext = realContexts.get(nbMoves + 1);
						realId = realIds.get(nbMoves + 1);
					}
					else {
						// We compute it otherwise
						realContext = new Context(realContext);
						realContext.game().apply(realContext, realMoves.get(nbMoves*2));
						// Apply the pass move
						// Shouldn't be a tie here
						realContext = new Context(realContext);
						realContext.game().apply(realContext, realContext.moves(realContext).get(0));
						realContexts.add(realContext);
						realId = createID(realContext);
						realIds.add(realId);
					}

					if (current.context.state().mover() == player ){
						// We're in a node corresponding to a move of the player that has already been played
						current = select(current, realMoves.get(nbMoves*2), realId, path);
					} else {
						// We're in a node corresponding to a move of the opponent but before the current state of the game
						current = select(current, null, realId, path);
					}
				} else {
					// We're in a node corresponding to after the current state of the game
					current = select(current, null, null, path);
				}
				
				if (current == null) {
					// We're in a node that is impossible
					break;
				}

				if (current.context.trial().over())
				{
					// We've reached a terminal state
					break;
				}
				
				if (current.visitCount == 0)
				{
					// We've expanded a new node, time for playout!
					break;
				}
				nbMoves++;
			}
			
			if (current != null){
				Context contextEnd = current.context;
				
				if (!contextEnd.trial().over())
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
				
				// This computes utilities for all players at the end of the playout,
				// which will all be values in [-1.0, 1.0]
				final double[] utilities = RankUtils.utilities(contextEnd);
				
				// Backpropagate utilities through the tree
				backPropagate(path, d1, d2, d3, utilities);
			}
			
			// Increment iteration count
			++numIterations;
		}

		System.out.println("number of iterations: " + numIterations);
		System.out.println("nb moves played: " + (realMoves.size() + 1)/2);
		// Return the only available action if there is only one (for example for a pass)
		if (context.game().moves(context).moves().size() == 1)
			return context.game().moves(context).moves().get(0);

		// Return the move we wish to play
		Move chosenMove = finalMoveSelection(context);
		return chosenMove;
	}
	
	/**
	 * Selects child of the given current node according to UCB1 equation.
	 * This method also implements the "Expansion" phase of MCTS, and creates
	 * a new node if the given current node has unexpanded moves.

	 * 
	 * @param current
	 * @return Selected node (if it has 0 visits, it will be a newly-expanded node).
	 */
	public Node select(final Node current, final Move realMove, ArrayList<Set<Integer>> idRealContext, Stack<Edge> path)
	{
		if (realMove != null){
			// We're in a node corresponding to a move of the player that has already been played so we expand only toward this move
			current.unexpandedMoves.clear();

			Boolean currentImpossible = false;
			Edge found = null;

			ArrayList<Set<Integer>> id2 = new ArrayList<Set<Integer>>();

			// If the node has already been explored, no need to create a new node
			for (java.util.Iterator<Edge> iterator = current.exitingEdges.iterator();  iterator.hasNext();  ){
				// Iterator and breaks because of concurrentModificationException
				Edge child = iterator.next();
				if (child.move.equals(realMove)){
					// We have the right move played so we return the corresponding child and delete the others, useless ones
					if (isCoherent(child.succ.id, idRealContext)){
						found = child;
					} else {
						currentImpossible = true;
					}
					break;
				}
				// It's not the right move played so we remove it
				iterator.remove();
			}
			
			if (found != null) {
				// We have the right move played so we return the corresponding child and delete the others, useless ones
				transpoTable.get(current.id).exitingEdges.removeAll(current.exitingEdges);
				current.exitingEdges.clear();
				current.exitingEdges.add(found);
				transpoTable.get(current.id).exitingEdges.add(found);
				path.push(found);

				return found.succ;
			}

			if (currentImpossible) {
				// unexpandedMoves is already empty
				transpoTable.get(current.id).exitingEdges.removeAll(current.exitingEdges);
				current.exitingEdges.clear();
				propagateImpossible(current);
				return null;
			}

			Context nextContext = new Context(current.context);
			nextContext.game().apply(nextContext, realMove);
			// Apply the pass move
			// In case of a tie there is no need to apply the pass move
			if (! nextContext.trial().over()){
				// Needs to recreate a context, else it crashes 
				nextContext = new Context(nextContext);
				nextContext.game().apply(nextContext, nextContext.game().moves(nextContext).moves().get(0));
			}
			id2 = createID(nextContext);

			if (isCoherent(idRealContext, id2)){
				Edge newEdge = new Edge(realMove, current, id2, nextContext);
				Node newNode = newEdge.succ;
				// We don't want to stop here so we add 1 to the visitCount
				newNode.visitCount = 1;
				newEdge.nd2 ++;
				newEdge.nd3 ++;
				newEdge.n ++;
				path.push(newEdge);

				transpoTable.get(current.id).exitingEdges.add(newEdge);

				return newNode;
			}
			else {
				System.out.println("real move not yet done impossible");
				// unexpandedMoves is already empty
				transpoTable.get(current.id).exitingEdges.removeAll(current.exitingEdges);
				current.exitingEdges.clear();
				propagateImpossible(current);
				return null;
			}
		}
		
		if (!current.unexpandedMoves.isEmpty())
		{
			// randomly select an unexpanded move
			final Move move = current.unexpandedMoves.remove(ThreadLocalRandom.current().nextInt(current.unexpandedMoves.size()));
			
			// create a copy of context
			Context context = new Context(current.context);
			
			// apply the move
			context.game().apply(context, move);
			// Apply the pass move
			// In case of a tie there is no need to apply the pass move
			if (! context.trial().over()){
				// Needs to recreate a context, else it crashes 
				context = new Context(context);
				context.game().apply(context, context.game().moves(context).moves().get(0));
			}

			//Compute id of the new context
			ArrayList<Set<Integer>> id2 = createID(context);

			if (idRealContext != null && ! (isCoherent(id2, idRealContext))){
				System.out.println("unexpanded move impossible");
				propagateImpossible(current);
				return null;
			}
			// create new node and return it
			Edge newEdge = new Edge(move, current, id2, context);
			transpoTable.get(current.id).exitingEdges.add(newEdge);
			Node newNode = newEdge.succ;
			path.push(newEdge);

			return newNode;
		}

		// use UCB1 equation to select from all children, with random tie-breaking
		Edge bestChild = null;
        double bestValue = Double.NEGATIVE_INFINITY;
		double pd2 = 0.0;

		// check if the node is impossible (happens, despite the cleaning with propagateImpossible, for some reason)
		if (current.exitingEdges.isEmpty()){
			System.out.println("Impossible node weirdly not deleted");
			System.exit(1);
		}

		Node combinedNode = transpoTable.get(current.id);

		for (Edge child : combinedNode.exitingEdges){
			pd2 += child.nd2;
		}
        final double twoParentLog = 2.0 * Math.log(Math.max(1, pd2));
        int numBestFound = 0;
        
        final int numChildren = combinedNode.exitingEdges.size();
        final int mover = current.context.state().mover();

		final HashMap<Move, double[]> scoreMeans = new HashMap<>();
		HashMap<Move, Integer> scorend3s = new HashMap<>();

		for (Edge e : combinedNode.exitingEdges){
			if (!scoreMeans.containsKey(e.move)){
				scoreMeans.put(e.move, e.scoreSum);
				scorend3s.put(e.move, e.nd3);
			}
			else {
				double [] scoreMeanSum = scoreMeans.get(e.move);
				for (int i=0; i<scoreMeanSum.length; i++){
					scoreMeanSum[i] = scoreMeanSum[i] +  e.scoreSum[i];
				}
				scoreMeans.put(e.move, scoreMeanSum);
				scorend3s.put(e.move, scorend3s.get(e.move) + e.nd3);
			}
		}

        for (int i = 0; i < numChildren; ++i) 
        {
        	final Edge child = combinedNode.exitingEdges.get(i);
        	final double exploit = child.scoreMean[mover];
        	final double explore = Math.sqrt(twoParentLog / child.nd3);
        
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
        
		path.push(bestChild);

		// We update the current context
		currentContext = new Context(currentContext);
		currentContext.game().apply(currentContext, bestChild.move);
		// Apply the pass move
		// In case of a tie there is no need to apply the pass move
		if (! currentContext.trial().over()){
			// Needs to recreate a context, else it crashes 
			currentContext = new Context(currentContext);
			currentContext.game().apply(currentContext, currentContext.moves(currentContext).get(0));
		}

        return bestChild.succ;
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
		int maxn = 0;
		Move bestMove = null;

		// Compute id of the context
		ArrayList<Set<Integer>> id = createID(context);

		// Check if the observation is in the transposition table
		if (!transpoTable.containsKey(id)){
			System.out.println(id);
			System.out.println("max depth reached: " + maxDepthReached);
			System.out.println("Error: context not in transposition table");
			FastArrayList<Move> moves = context.game().moves(context).moves();
			return moves.get(ThreadLocalRandom.current().nextInt(moves.size()));
		}

		for (Edge edge : transpoTable.get(id).key().exitingEdges){
			if (edge.succ.visitCount > maxn){
				maxn = edge.succ.visitCount;
				bestMove = edge.move;
			} else  if (edge.succ.visitCount == maxn){
				if (ThreadLocalRandom.current().nextInt() % 2 == 0){
					bestMove = edge.move;
				}
			}
		}

		if (bestMove == null){
			FastArrayList<Move> moves = context.game().moves(context).moves();
			return moves.get(ThreadLocalRandom.current().nextInt(moves.size()));
		}

		return bestMove;
	}

	private boolean isCoherent(ArrayList<Set<Integer>> idContext, ArrayList<Set<Integer>> idPredictedContext){
		if (idContext.equals(idPredictedContext)){
			return true;
		}
		else {
			return false;
		}
	}

	private void propagateImpossible(Node parent){
		System.out.println("propagate impossible depth : " + parent.depth);
		System.out.println("parent : " + parent);
		if (!parent.unexpandedMoves.isEmpty()){
			return;
		}

		if (parent.exitingEdges.isEmpty()){
			// No unexpanded moves normally
			transpoTable.remove(parent.id);
			if (!(parent.enteringEdges.size()==1 && parent.enteringEdges.get(0) == null)) {
				for (Edge edge : parent.enteringEdges){
					System.out.println("exiting edges : " + edge.pred.exitingEdges);
					System.out.println("edge : " + edge);
					edge.pred.exitingEdges.remove(edge);
					System.out.println("En chaines");
					propagateImpossible(edge.pred);
				}
				// parent.enteringEdges.clear();
			}
		}
		return;
	}

	private void backPropagate(Stack<Edge> path, int d1, int d2, int d3, double[] results){
		System.out.println("backpropagate");
		Edge leaf = path.peek();
		System.out.println("depth : " + leaf.succ.depth);
		Set<Edge> toClearDelta = new HashSet<Edge>();
		toClearDelta.addAll(path);
		Set<Edge> nextLayer = new HashSet<Edge>();
		nextLayer.add(leaf);
		int max = Math.max(d1, Math.max(d2, d3));
		for (int i=0; i<=max && !path.empty(); i++){
			Set<Edge> edges = new HashSet<>(nextLayer);
			nextLayer.clear();
			for (Edge edge : edges){
				toClearDelta.add(edge);

				if (!(edge.pred.enteringEdges.size()==1 && edge.pred.enteringEdges.get(0) == null)) {
					nextLayer.addAll(edge.pred.enteringEdges);
				}

				// No need to update twice
				if (edge == path.peek()){
					path.pop();
					edge.nd2 ++;
					edge.nd3 ++;
					updateMean(edge, i, results);
					edge.n ++;
					edge.succ.visitCount ++;
					continue;
				}

				if (i <= d2){
					edge.nd2 ++;
				}
				if (i <= d3){
					edge.nd3 ++;
				}
				if (i <= d1){
					updateMean(edge, i, results);
				}
			}
		}
		while (!path.empty())
		{
			Edge edge = path.pop();
			edge.nd2 ++;
			edge.nd3 ++;
			updateMean(edge, max, results);
			edge.n ++;
			edge.succ.visitCount ++;
		}

		for (Edge edge : toClearDelta){
			edge.deltaMean = new double[edge.scoreMean.length];
		}

		return ;
	}
	
	/**
	 * Update the mean of the score of the edge 
	 * <p>
	 * /!\ nd3 need to have been updated prior to calling this function
	 * 
	 * @param edge
	 * @param i (depth of the node)
	 * @param results
	 */
	private void updateMean(Edge edge, int i, double[] results){
		for (int j=0; j<edge.deltaMean.length; j++){
			if (i==0){
				// nd3 has already been incremented
				double value = (edge.scoreMean[j] * (edge.nd3 - 1) + results[j]) / edge.nd3;
				edge.deltaMean[j] = value - edge.scoreMean[j];
				edge.scoreMean[j] = value;
			}
			else {
				int n_tot = 0;
				edge.deltaMean[j] = 0.0;
				for (Edge child_edge : edge.succ.exitingEdges){
					edge.deltaMean[j] += child_edge.deltaMean[j] * child_edge.n;
					n_tot += child_edge.n;
				}
				edge.deltaMean[j] /= n_tot + edge.n_prime;
				edge.scoreMean[j] += edge.deltaMean[j];
			}
		}
		return ;
	}

	private ArrayList<Set<Integer>> createID(Context context){
		TIntArrayList ownedSites = context.state().owned().sites(player);
		Set<Integer> ownedSet = new HashSet<>(ownedSites.size());
		for (int i = 0; i < ownedSites.size(); i++) {
			ownedSet.add(ownedSites.get(i));
		}
		ArrayList<Set<Integer>> id = new ArrayList<Set<Integer>>();
		id.add(ownedSet);
		HashSet<Integer> moveNumber = new HashSet<>();
		moveNumber.add(context.trial().moveNumber());
		id.add(moveNumber);
		return id;
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
	 * Inner class for nodes used by UCD
	 * 
	 * @author Aymeric Behaegel
	 */
	private class Node
	{
        /** Entering edges **/
        private final List<Edge> enteringEdges = new ArrayList<Edge>();

		/** Exiting Edges */
		private final List<Edge> exitingEdges = new ArrayList<Edge>();
		
		/** The id of the observation of this Node for the transposition table */
		private ArrayList<Set<Integer>> id;

		/** Context of the Node */
		private Context context;

		/** Visit count for this node */
		private int visitCount = 0;

		/** Depth of the node in the tree */
		private int depth = 0;
		
		/** List of moves for which we did not yet create a child node */
		private final FastArrayList<Move> unexpandedMoves;


		/**
		 * Constructor
		 * 
		 * @param parent
		 * @param moveFromParent
		 * @param context
		 * @param id
		 */
		public Node(final Edge edge, final Context context, ArrayList<Set<Integer>> id)
		{
			this.enteringEdges.add(edge);
			if (edge != null){
				depth = edge.pred.depth + 1;
				if (depth > maxDepthReached){
					maxDepthReached = depth;
				}
			}

			this.id = id;

			if (transpoTable.containsKey(id)){
				transpoTable.get(id).enteringEdges.add(edge);
			} else {
				transpoTable.put(id, new Node(edge, context, id));
			}

			// For simplicity, we just take ALL legal moves. 
			// This means we do not support simultaneous-move games.
			unexpandedMoves = new FastArrayList<Move>(context.game().moves(context).moves());
			
			this.context = context;
		}
	}

    /**
	 * Inner class for edges used by UCD
	 * 
	 * @author Aymeric Behaegel
	 */
	private class Edge
    {
        /** The move that this edge represent */
        private final Move move;
        
        /** The node that is at the beginning of this edge */
        private final Node pred;

        /** The node that is at the end of this edge */
        private final Node succ;

		/** The context corresponding to the  */

		/** For every player, mean of utilities / scores backpropagated through this node */
		private double[] scoreMean;

		/** For every player, sum of utilities / scores backpropagated through this node */
		private double[] scoreSum;

		/** Variation of scoreMean; used for updating scoreMean */
		private double[] deltaMean;

		/** Simple visit count */
		private int n = 0;

		/** Visit count before first child is created */
		private int n_prime = 0;

        /** Visit count extended for calculating pd2 */
		private int nd2 = 0;
		
		/** Visit count extended for this edge */
        private int nd3 = 0;
        

		/**
         * Constructor
         * 
         * @param move
         * @param pred
         * @param succ
		 * @param id
		 * @param contextSucc
         */
        public Edge(final Move move, final Node pred, ArrayList<Set<Integer>> id, Context contextSucc)
        {
            this.move = move;
            this.pred = pred;

			// Set n_prime for the pred node (i.e. all entering edges) if it is the first time we create a child
			if (pred.exitingEdges.isEmpty() && !(pred.enteringEdges.size()==1 && pred.enteringEdges.get(0) == null)){
				for (Edge edge : pred.enteringEdges){
					edge.n_prime = edge.n;
				}
			}

            pred.exitingEdges.add(this);

            this.succ = new Node(this, contextSucc, id);

			transpoTable.get(pred.id).exitingEdges.add(this);

            this.scoreMean = new double[contextSucc.game().players().count() + 1];
			this.deltaMean = new double[contextSucc.game().players().count() + 1];
			this.scoreSum = new double[contextSucc.game().players().count() + 1];
            this.nd2 = 0;
            this.nd3 = 0;
        }
    }

	//-------------------------------------------------------------------------

}
