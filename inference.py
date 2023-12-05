# inference.py
# ------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import itertools
import random
import busters
import game

from util import manhattanDistance, raiseNotDefined, Counter


class DiscreteDistribution(dict):
    """
    A DiscreteDistribution models belief distributions and weight distributions
    over a finite set of discrete keys.
    """
    def __getitem__(self, key):
        self.setdefault(key, 0)
        return dict.__getitem__(self, key)

    def copy(self):
        """
        Return a copy of the distribution.
        """
        return DiscreteDistribution(dict.copy(self))

    def argMax(self):
        """
        Return the key with the highest value.
        """
        if len(self.keys()) == 0:
            return None
        all = list(self.items())
        values = [x[1] for x in all]
        maxIndex = values.index(max(values))
        return all[maxIndex][0]

    def total(self):
        """
        Return the sum of values for all keys.
        """
        return float(sum(self.values()))

    def normalize(self):
        """
        Normalize the distribution such that the total value of all keys sums
        to 1. The ratio of values for all keys will remain the same. In the case
        where the total value of the distribution is 0, do nothing.
        >>> dist = DiscreteDistribution()
        >>> dist['a'] = 1
        >>> dist['b'] = 2
        >>> dist['c'] = 2
        >>> dist['d'] = 0
        >>> dist.normalize()
        >>> list(sorted(dist.items()))
        [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0)]
        >>> dist['e'] = 4
        >>> list(sorted(dist.items()))
        [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0), ('e', 4)]
        >>> empty = DiscreteDistribution()
        >>> empty.normalize()
        >>> empty
        {}
        """
        "*** YOUR CODE HERE ***"
        total = float(self.total()) #get total 
        if total == 0: return #if there's nothing in there do nothing
        for key in self.keys(): 
            self[key] = self[key] / total #divide each key by the total, getting each of their proportions
        return
        raiseNotDefined()

    def sample(self):
        """
        Draw a random sample from the distribution and return the key, weighted
        by the values associated with each key.

        >>> dist = DiscreteDistribution()
        >>> dist['a'] = 1
        >>> dist['b'] = 2
        >>> dist['c'] = 2
        >>> dist['d'] = 0
        >>> N = 100000.0
        >>> samples = [dist.sample() for _ in range(int(N))]
        >>> round(samples.count('a') * 1.0/N, 1)  # proportion of 'a'
        0.2
        >>> round(samples.count('b') * 1.0/N, 1)
        0.4
        >>> round(samples.count('c') * 1.0/N, 1)
        0.4
        >>> round(samples.count('d') * 1.0/N, 1)
        0.0
        """
        "*** YOUR CODE HERE ***"
        curr = 0.0 #initialize lower limit at zero
        picker = random.random() #random number from 0-1
        self.normalize() #normalize, so everything sums to 1
        for key in self.keys():
            if curr <= picker and picker < self[key] + curr: #if the key falls in a block the size of teh key's proportion, return the key
                return key
            curr += self[key]
        return
        raiseNotDefined()


class InferenceModule:
    """
    An inference module tracks a belief distribution over a ghost's location.
    """
    ############################################
    # Useful methods for all inference modules #
    ############################################

    def __init__(self, ghostAgent):
        """
        Set the ghost agent for later access.
        """
        self.ghostAgent = ghostAgent
        self.index = ghostAgent.index
        self.obs = []  # most recent observation position

    def getJailPosition(self):
        return (2 * self.ghostAgent.index - 1, 1)

    def getPositionDistributionHelper(self, gameState, pos, index, agent):
        try:
            jail = self.getJailPosition()
            gameState = self.setGhostPosition(gameState, pos, index + 1)
        except TypeError:
            jail = self.getJailPosition(index)
            gameState = self.setGhostPositions(gameState, pos)
        pacmanPosition = gameState.getPacmanPosition()
        ghostPosition = gameState.getGhostPosition(index + 1)  # The position you set
        dist = DiscreteDistribution()
        if pacmanPosition == ghostPosition:  # The ghost has been caught!
            dist[jail] = 1.0
            return dist
        pacmanSuccessorStates = game.Actions.getLegalNeighbors(pacmanPosition, \
                gameState.getWalls())  # Positions Pacman can move to
        if ghostPosition in pacmanSuccessorStates:  # Ghost could get caught
            mult = 1.0 / float(len(pacmanSuccessorStates))
            dist[jail] = mult
        else:
            mult = 0.0
        actionDist = agent.getDistribution(gameState)
        for action, prob in actionDist.items():
            successorPosition = game.Actions.getSuccessor(ghostPosition, action)
            if successorPosition in pacmanSuccessorStates:  # Ghost could get caught
                denom = float(len(actionDist))
                dist[jail] += prob * (1.0 / denom) * (1.0 - mult)
                dist[successorPosition] = prob * ((denom - 1.0) / denom) * (1.0 - mult)
            else:
                dist[successorPosition] = prob * (1.0 - mult)
        return dist

    def getPositionDistribution(self, gameState, pos, index=None, agent=None):
        """
        Return a distribution over successor positions of the ghost from the
        given gameState. You must first place the ghost in the gameState, using
        setGhostPosition below.
        """
        if index == None:
            index = self.index - 1
        if agent == None:
            agent = self.ghostAgent
        return self.getPositionDistributionHelper(gameState, pos, index, agent)

    def getObservationProb(self, noisyDistance, pacmanPosition, ghostPosition, jailPosition):
        """
        Return the probability P(noisyDistance | pacmanPosition, ghostPosition).
        """
        "*** YOUR CODE HERE ***"
        if noisyDistance == None and ghostPosition == jailPosition: #None and jail = 1.0 probability
            return 1.0
        if noisyDistance == None and ghostPosition != jailPosition: #None and not jail = zero chance
            return 0.0
        if noisyDistance != None and ghostPosition == jailPosition: #not none and jail = zero chance
            return 0.0
        else: #otherwise use the function they give you
            manhattan = manhattanDistance(pacmanPosition, ghostPosition)
            return busters.getObservationProbability(noisyDistance, manhattan) #P(noisyDistance | manhattan)
        raiseNotDefined()

    def setGhostPosition(self, gameState, ghostPosition, index):
        """
        Set the position of the ghost for this inference module to the specified
        position in the supplied gameState.

        Note that calling setGhostPosition does not change the position of the
        ghost in the GameState object used for tracking the true progression of
        the game.  The code in inference.py only ever receives a deep copy of
        the GameState object which is responsible for maintaining game state,
        not a reference to the original object.  Note also that the ghost
        distance observations are stored at the time the GameState object is
        created, so changing the position of the ghost will not affect the
        functioning of observe.
        """
        conf = game.Configuration(ghostPosition, game.Directions.STOP)
        gameState.data.agentStates[index] = game.AgentState(conf, False)
        return gameState

    def setGhostPositions(self, gameState, ghostPositions):
        """
        Sets the position of all ghosts to the values in ghostPositions.
        """
        for index, pos in enumerate(ghostPositions):
            conf = game.Configuration(pos, game.Directions.STOP)
            gameState.data.agentStates[index + 1] = game.AgentState(conf, False)
        return gameState

    def observe(self, gameState):
        """
        Collect the relevant noisy distance observation and pass it along.
        """
        distances = gameState.getNoisyGhostDistances()
        if len(distances) >= self.index:  # Check for missing observations
            obs = distances[self.index - 1]
            self.obs = obs
            self.observeUpdate(obs, gameState)
        
    def initialize(self, gameState):
        """
        Initialize beliefs to a uniform distribution over all legal positions.
        """
        self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
        self.allPositions = self.legalPositions + [self.getJailPosition()]
        self.initializeUniformly(gameState)

    ######################################
    # Methods that need to be overridden #
    ######################################

    def initializeUniformly(self, gameState):
        """
        Set the belief state to a uniform prior belief over all positions.
        """
        raise NotImplementedError

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the given distance observation and gameState.
        """
        raise NotImplementedError

    def elapseTime(self, gameState):
        """
        Predict beliefs for the next time step from a gameState.
        """
        raise NotImplementedError

    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence so far.
        """
        raise NotImplementedError


class ExactInference(InferenceModule):
    """
    The exact dynamic inference module should use forward algorithm updates to
    compute the exact belief function at each time step.
    """
    def initializeUniformly(self, gameState):
        """
        Begin with a uniform distribution over legal ghost positions (i.e., not
        including the jail position).
        """
        self.beliefs = DiscreteDistribution()
        for p in self.legalPositions:
            self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distance to the ghost you are
        tracking.

        self.allPositions is a list of the possible ghost positions, including
        the jail position. You should only consider positions that are in
        self.allPositions.

        The update model is not entirely stationary: it may depend on Pacman's
        current position. However, this is not a problem, as Pacman's current
        position is known.
        """

        "*** YOUR CODE HERE ***"    
        pacmanPosition = gameState.getPacmanPosition()
        update = DiscreteDistribution() #new discrete distribution to store stuff

        for position in self.allPositions: #for each possible position
            if self.getObservationProb(observation, pacmanPosition, position, self.getJailPosition()) != 0: #if there's a nonzero chance a ghost is there
                #set it equal to old belief * probability you get that reading is correct given the ghost is there
                update[position] = self.beliefs[position] * self.getObservationProb(observation, pacmanPosition, position, self.getJailPosition())
        #normalize update, and replace beliefs with it
        update.normalize()
        self.beliefs = update
        return
        raiseNotDefined()

        self.beliefs.normalize()

    def elapseTime(self, gameState):
        """
        Predict beliefs in response to a time step passing from the current
        state.

        The transition model is not entirely stationary: it may depend on
        Pacman's current position. However, this is not a problem, as Pacman's
        current position is known.
        """
        "*** YOUR CODE HERE ***"
        #questions:
        #is gameState the current gameState, or the previous one?
        
        update = DiscreteDistribution() #initialize new discrete distribution
        #for every old position
        for oldPos in self.allPositions:
            #if it's possible for a ghost to be there
            if self.beliefs[oldPos] > 0:
                #get the new distribution for all possible new positions
                newPosDist = self.getPositionDistribution(gameState, oldPos)
                for newPos in newPosDist:
                    #update belief for each new position, based on your belief it can be there and the odds it goes from previous state to newPos
                    update[newPos] = update[newPos] + (newPosDist[newPos] * self.beliefs[oldPos])
        #normalize update and replace beliefs
        update.normalize()
        self.beliefs = update
        return
        raiseNotDefined()

    def getBeliefDistribution(self):
        return self.beliefs


class ParticleFilter(InferenceModule):
    """
    A particle filter for approximately tracking a single ghost.
    """
    def __init__(self, ghostAgent, numParticles=300):
        InferenceModule.__init__(self, ghostAgent)
        self.setNumParticles(numParticles)

    def setNumParticles(self, numParticles):
        self.numParticles = numParticles

    def initializeUniformly(self, gameState):
        """
        Initialize a list of particles. Use self.numParticles for the number of
        particles. Use self.legalPositions for the legal board positions where
        a particle could be located. Particles should be evenly (not randomly)
        distributed across positions in order to ensure a uniform prior. Use
        self.particles for the list of particles.
        """
        self.particles = []
        "*** YOUR CODE HERE ***"
        #index to keep track of where you are in the list of legal positions
        index = 0
        for i in range(0, self.numParticles): #until you fill up particles with numParticles particles
            if index < len(self.legalPositions):
                self.particles.append(self.legalPositions[index]) #add the next legal position
            else: #if you go over, set index to zero and restart
                index = 0
                self.particles.append(self.legalPositions[index])
            index = index + 1
        return self.particles 
        raiseNotDefined()

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distance to the ghost you are
        tracking.

        There is one special case that a correct implementation must handle.
        When all particles receive zero weight, the list of particles should
        be reinitialized by calling initializeUniformly. The total method of
        the DiscreteDistribution may be useful.
        """
        "*** YOUR CODE HERE ***"
        beliefs = self.getBeliefDistribution() #get beliefs
        weighted = DiscreteDistribution() #initialize weighted discrete distribution
        newParticles = [] #and new list of particles

        for i in range(0, self.numParticles):
            newParticle = beliefs.sample() #sample from beliefs to get a particle

            newParticles.append(newParticle) #and add to newParticles
            weighted[newParticle] += self.getObservationProb(observation, gameState.getPacmanPosition(), newParticle, self.getJailPosition()) #then update the weight for it
        
        if weighted.total() == 0: #if all weights are zero, reinitialize everything from your current gameState
            self.initializeUniformly(gameState)
            return
        
        weighted.normalize() #normalize weighted dd
        newNewParticles = [] #get new, new particles list
        for i in range(0,len(newParticles)): #sample from weighted distribution until you fill up newnewparticles
            newNewParticles.append(weighted.sample())
        self.particles = newNewParticles #update self.particles
        return newNewParticles
        raiseNotDefined()

    def elapseTime(self, gameState):
        """
        Sample each particle's next state based on its current state and the
        gameState.
        """
        "*** YOUR CODE HERE ***"
        #get beliefs and initialize weighted dd
        updateWeights = DiscreteDistribution() 
        beliefs = self.getBeliefDistribution()
        #for every old position
        for oldPos in self.allPositions:
            #if you believe it's possible for you to have been there
            if beliefs[oldPos] > 0:
                #get distribution of new possible states
                newPosDist = self.getPositionDistribution(gameState, oldPos)
                #and update the weighted distribution
                for newPos in newPosDist:
                    updateWeights[newPos] = updateWeights[newPos] + (newPosDist[newPos] * beliefs[oldPos])
        updateWeights.normalize() #normalize updated weights
        newParticles = [] #and initialize newParticles
        for i in range(0, self.numParticles): #sample numParticles particles from weighted distribution
            newParticles.append(updateWeights.sample())
        self.particles = newParticles #update self.particles
        return newParticles
        raiseNotDefined()

    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence and time passage. This method
        essentially converts a list of particles into a belief distribution.
        
        This function should return a normalized distribution.
        """
        "*** YOUR CODE HERE ***"
        belief = DiscreteDistribution() #set up discrete distribution
        for particle in self.particles: #add 1 to the value for every time the key occurs
            belief[particle] += 1
        belief.normalize() #normalize it
        return belief 
        raiseNotDefined()


class JointParticleFilter(ParticleFilter):
    """
    JointParticleFilter tracks a joint distribution over tuples of all ghost
    positions.
    """
    def __init__(self, numParticles=600):
        self.setNumParticles(numParticles)

    def initialize(self, gameState, legalPositions):
        """
        Store information about the game, then initialize particles.
        """
        self.numGhosts = gameState.getNumAgents() - 1
        self.ghostAgents = []
        self.legalPositions = legalPositions
        self.initializeUniformly(gameState)

    def initializeUniformly(self, gameState):
        """
        Initialize particles to be consistent with a uniform prior. Particles
        should be evenly distributed across positions in order to ensure a
        uniform prior.
        """
        self.particles = []
        "*** YOUR CODE HERE ***"

        np = self.numParticles
        bigProductOhBoy = list(itertools.product(self.legalPositions, repeat=self.numGhosts)) #make legalPositions^self.numGhosts mega list of positions
        random.shuffle(bigProductOhBoy) #and shuffle the big list
        index = 0 #index to keep track
        while index < np: #while there are less than numParticles
            for particle in bigProductOhBoy:
                if index < np:
                    self.particles.append(particle) #fill it up with a particle from the shuffled big list until it's full
                    index += 1
                else:
                    break
        return self.particles



        raiseNotDefined()

    def addGhostAgent(self, agent):
        """
        Each ghost agent is registered separately and stored (in case they are
        different).
        """
        self.ghostAgents.append(agent)

    def getJailPosition(self, i):
        return (2 * i + 1, 1)

    def observe(self, gameState):
        """
        Resample the set of particles using the likelihood of the noisy
        observations.
        """
        observation = gameState.getNoisyGhostDistances()
        self.observeUpdate(observation, gameState)

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distances to all ghosts you
        are tracking.

        There is one special case that a correct implementation must handle.
        When all particles receive zero weight, the list of particles should
        be reinitialized by calling initializeUniformly. The total method of
        the DiscreteDistribution may be useful.
        """
        "*** YOUR CODE HERE ***"
        
        pacmanPosition = gameState.getPacmanPosition()
        noisyDistances = observation
        # if len(noisyDistances) < self.numGhosts: return #if there are less noisy readings than ghosts, return

        update = DiscreteDistribution() #initialize new discrete distribution
        for particle in self.particles: #for every particle in self.particles
            val = 1.0 #initialize value to be 1
            for i in range(self.numGhosts):
                #for each ghost, multiply by the probability
                val *= self.getObservationProb(noisyDistances[i], pacmanPosition, particle[i], self.getJailPosition(i))
            #and add to the updated value
            update[particle] += val
        
        if update.total() == 0: #if total is zero, reinitialize from current gameState
            self.initializeUniformly(gameState)
            return

        update.normalize() #normalize update
        newParticles = []
        for _ in range(0, self.numParticles): #and sample numParticles from it for newParticles
            newParticles.append(update.sample())
        self.particles = newParticles #return newParticles
        return
                
        raiseNotDefined()

    def elapseTime(self, gameState):
        """
        Sample each particle's next state based on its current state and the
        gameState.
        """
        newParticles = []
        for oldParticle in self.particles:
            newParticle = list(oldParticle)  # A list of ghost positions

            # now loop through and update each entry in newParticle...
            "*** YOUR CODE HERE ***"

            for i in range(self.numGhosts): #for every ghost
                #get distribution of possible new positions, given positions of all ghosts
                newPosDist = self.getPositionDistribution(gameState, oldParticle, i, self.ghostAgents[i])
                #and newParticle[i] to be some sampled location from distribution
                newParticle[i] = newPosDist.sample()

            # raiseNotDefined()

            """*** END YOUR CODE HERE ***"""
            newParticles.append(tuple(newParticle))
        self.particles = newParticles


# One JointInference module is shared globally across instances of MarginalInference
jointInference = JointParticleFilter()


class MarginalInference(InferenceModule):
    """
    A wrapper around the JointInference module that returns marginal beliefs
    about ghosts.
    """
    def initializeUniformly(self, gameState):
        """
        Set the belief state to an initial, prior value.
        """
        if self.index == 1:
            jointInference.initialize(gameState, self.legalPositions)
        jointInference.addGhostAgent(self.ghostAgent)

    def observe(self, gameState):
        """
        Update beliefs based on the given distance observation and gameState.
        """
        if self.index == 1:
            jointInference.observe(gameState)

    def elapseTime(self, gameState):
        """
        Predict beliefs for a time step elapsing from a gameState.
        """
        if self.index == 1:
            jointInference.elapseTime(gameState)

    def getBeliefDistribution(self):
        """
        Return the marginal belief over a particular ghost by summing out the
        others.
        """
        jointDistribution = jointInference.getBeliefDistribution()
        dist = DiscreteDistribution()
        for t, prob in jointDistribution.items():
            dist[t[self.index - 1]] += prob
        return dist
