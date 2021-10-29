import ecole as ec
import numpy as np
import pyscipopt


class ObservationFunction():

    def __init__(self, problem):
        # called once for each problem benchmark
        self.problem = problem  # to devise problem-specific observations
        self.m_copy = None

    def seed(self, seed):
        # called before each episode
        # use this seed to make your code deterministic
        pass

    def before_reset(self, model):
        # called when a new episode is about to start
        self.m_copy = None

    def extract(self, model, done):
        if done:
            return None

        m = model.as_pyscipopt()
        remaining_time_budget = m.getParam("limits/time") - m.getSolvingTime()

        # extract (copy) the model only once
        if self.m_copy is None:
            self.m_copy = pyscipopt.Model(sourceModel=m)  # copy the model into a new SCIP instance
            m_isfresh = True  # freshly created model
        else:
            m_isfresh = False  # not freshly created model

        observation = (self.m_copy, remaining_time_budget, m_isfresh)

        return observation


class Policy():

    def __init__(self, problem):
        # called once for each problem benchmark
        self.rng = np.random.RandomState()
        self.problem = problem  # to devise problem-specific policies
        
        #set to some values just to make sure
        self.heuristics = pyscipopt.SCIP_PARAMSETTING.AGGRESSIVE
        self.separating = pyscipopt.SCIP_PARAMSETTING.DEFAULT
        self.emphasis = pyscipopt.SCIP_PARAMEMPHASIS.DEFAULT

        if problem == 'item_placement':
            self.schedule = 'None'
            self.heuristics = pyscipopt.SCIP_PARAMSETTING.AGGRESSIVE
            self.separating = pyscipopt.SCIP_PARAMSETTING.DEFAULT
            self.emphasis = pyscipopt.SCIP_PARAMEMPHASIS.FEASIBILITY

        elif problem == 'load_balancing':
            #use 2h schedule for 2nd dataset
            self.schedule = 'schedules/schedule2_30min.set'
            self.heuristics = pyscipopt.SCIP_PARAMSETTING.FAST
            self.separating = pyscipopt.SCIP_PARAMSETTING.OFF
            self.emphasis = pyscipopt.SCIP_PARAMEMPHASIS.PHASEFEAS

        elif problem == 'anonymous':
            self.schedule = 'schedules/schedule3_2h.set'
            self.heuristics = pyscipopt.SCIP_PARAMSETTING.AGGRESSIVE
            self.separating = pyscipopt.SCIP_PARAMSETTING.OFF
            self.emphasis = pyscipopt.SCIP_PARAMEMPHASIS.FEASIBILITY

        else:
            raise ValueError('Do not know any schedule for this problem.')

    def seed(self, seed):
        # called before each episode
        # use this seed to make your code deterministic
        self.rng = np.random.RandomState(seed)

    def __call__(self, action_set, observation):
        m, max_time_budget, m_isfresh = observation
        var_ids = action_set

        # reset solution improvement and solving time counters for freshly created models
        if m_isfresh:
            self.sol_improvements_consumed = 0
            self.solving_time_consumed = 0

        m.setPresolve(pyscipopt.scip.PY_SCIP_PARAMSETTING.OFF)  # presolve has already been done

        m.setHeuristics(self.heuristics)  # we want to focus on finding feasible solutions
        m.setSeparating(self.separating)
        m.setEmphasis(self.emphasis)

        if not (self.schedule == 'None'):
            m.readParams(self.schedule)

        m.setParam('limits/bestsol', self.sol_improvements_consumed + 1)  # stop after a new primal bound improvement is found
        m.setParam('limits/time', self.solving_time_consumed + max(max_time_budget - 0.01, 0))  # stop the agent before the environment times out
        m.setParam('limits/memory', 12000)
        m.optimize()

        # keep track of best sol improvements in the copied model to be able to set a limit
        if m.getStatus() == 'bestsollimit':
            self.sol_improvements_consumed += 1

        # keep track of solving time already spedn in the model to be able to increment it appropriately
        self.solving_time_consumed = m.getSolvingTime()

        if not m.getNSols() > 0:
            print('No solution was found. Return something trivial to make sure we do not error out.')
            # we could improve this slightly probably..
            sol_vals = np.asarray([0.0 for var in m.getVars(transformed=False)])
            action = (var_ids, sol_vals[var_ids])
        else:
            #print(f"{m.getSolvingTime()} seconds spend searching, best solution so far {m.getObjVal()}")
            sol = m.getBestSol()
            sol_vals = np.asarray([m.getSolVal(sol, var) for var in m.getVars(transformed=False)])
            action = (var_ids, sol_vals[var_ids])

        return action