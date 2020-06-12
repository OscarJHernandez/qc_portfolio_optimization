import cirq
import sympy
import numpy as np
import itertools
from scipy.optimize import minimize
from tqdm import tqdm
import matplotlib.pyplot as plt
import json


class Portfolio():

    def __init__(self, N_portfolio=0, mu=None, sigma=None,lam=None):
        '''

        '''

        # The number of items in the portfolio
        self.N_portfolio = N_portfolio

        # The number of quibits required is twice the number of items
        self.N_qubits = 2*N_portfolio

        # Initialize the quibits that will be used to represent the portfolio
        self.qubits = [cirq.LineQubit(i) for i in range(self.N_qubits)]

        # The indices required for this problem
        # (portfolio index, s^+_i index, s^-_i index
        self.portfolio_indices = [(i,2*i,2*i+1) for i in range(N_portfolio)]

        # Instantiate the returns vector
        self.mu = sympy.symbols(["mu_"+str(k) for k in range(N_portfolio)])

        # instantiate the sigma matrix
        self.sigma = sympy.symbols([["sigma_"+str(i)+str(j) for i in range(N_portfolio)] for j in range(N_portfolio)])

        # instantiate the Lambda value
        self.lam = sympy.symbols("L")

        # Instantiate the transaction cost
        self.T = sympy.symbols("T")

        # The constraint function
        self.D = sympy.symbols("D")

        # instantiate the previous weights for the portfolio
        self.y = sympy.symbols(["y_"+str(k) for k in range(N_portfolio)])

        return None

    def benchmark_values(self):
        '''
        The returns and covariance matrix for the stocks:

        AMP, ANZ, BHP, BXB, CBA, CSL, IAG, TLS

        from 2018,
        '''


        mu = np.array([0.000401, 0.000061, 0.000916,-0.000619,0.000212, 0.001477, 0.001047,-0.000881])

        sigma = np.array([[99.8,42.5, 37.2,40.3,38.0,30.0,46.8,14.9],
                  [42.5, 100.5, 41.1, 15.2, 71.1, 27.8, 47.5, 12.7],
                  [37.2, 41.1, 181.3, 17.9, 38.4, 27.9, 39.0, 8.3],
                  [40.3, 15.2, 17.9, 253.1, 12.4, 48.7, 33.3, 3.8],
                  [38.0, 71.1, 38.4, 12.4,84.7, 28.5, 42.0, 13.1],
                  [30.0, 27.8, 27.9, 48.7, 28.5, 173.1, 28.9, -12.7],
                  [46.8, 47.5, 39.0, 33.3, 42.0, 28.9, 125.8, 14.6],
                  [14.9, 12.7, 8.3, 3.8, 13.1, -12.7, 14.6, 179.0]])

        sigma = sigma/10**6

        # The constraint chosen in the paper
        if(self.N_portfolio%2 == 0):
            D = self.N_portfolio/2
        else:
            D = (self.N_portfolio+1)/2

        lam = 0.9
        T = 0
        A = 0.03
        y = np.zeros(len(mu))

        parameters={}


        parameters['mu'] = mu[0:self.N_portfolio]
        parameters['sigma'] = sigma[0:self.N_portfolio,0:self.N_portfolio]
        parameters['D'] = D
        parameters['lam'] = lam
        parameters['T'] = T
        parameters['A'] = A
        parameters['y'] = y



        return parameters


    def brute_force_search(self,parameters):
        '''
        This function determines the optimal solutions by brute force.
        '''

        results={}


        # Retrieve the values of the needed parameters
        lam = parameters['lam']
        D= parameters['D']
        mu = parameters['mu']
        sigma=parameters['sigma']
        T=parameters['T']
        y=parameters['y']

        best_solutions = None

        # Search the space of feasible solutions
        # The values that can be taken by Zi
        index = [-1,0, 1]

        keys = list(itertools.product(index, repeat=self.N_portfolio))

        # filter out non-feasable solutions
        feasible = []

        for key in keys:
            z = np.array(key)
            sum = np.sum(z)

            if(sum==D):
                feasible.append(z)

        feasible = np.array(feasible)
        state_costs = np.zeros(len(feasible))


        # Find the best solutions and also the worst solutions
        for k in range(len(feasible)):
            state = feasible[k]
            portfolio_cost = self.compute_portfolio_cost(lam, mu, sigma, state)
            transaction_cost = self.compute_transaction_cost(T,y,state)
            state_costs[k] = portfolio_cost+transaction_cost

        max_cost_indx = np.argwhere(state_costs == np.amax(state_costs)).flatten()
        min_cost_indx = np.argwhere(state_costs == np.amin(state_costs)).flatten()

        results['lambda'] = lam
        results['y'] = y
        results['mu'] = mu
        results['sigma'] = sigma
        results['D'] = D

        results['maximum_cost_states'] = feasible[max_cost_indx]
        results['maximum_cost'] = max(state_costs)
        results['volatility_of_maximum_cost_state'] = self.compute_portfolio_volatility(sigma, feasible[max_cost_indx][0])
        results['returns_of_maximum_cost_state'] = self.compute_portfolio_returns(mu, feasible[max_cost_indx][0])


        results['minimum_cost_states'] = feasible[min_cost_indx]
        results['minimum_cost'] = min(state_costs)
        results['volatility_of_minimum_cost_state'] = self.compute_portfolio_volatility(sigma, feasible[min_cost_indx][0])
        results['returns_of_minimum_cost_state'] = self.compute_portfolio_returns(mu, feasible[min_cost_indx][0])

        return results

    def QAOA_circuit(self,p=1,betas=1,gammas=1):
        '''
        Construct the QAOA circuit with soft constraints
        '''

        # Instantiate the circuit with the symbolic parameters that we will need
        self.gammas = sympy.symbols(["gamma_" + str(k) for k in range(p)])
        self.betas = sympy.symbols(["beta_" + str(k) for k in range(p)])

        circuit = cirq.Circuit()
        circuit = self.apply_hadamard(circuit)

        for k in range(0,p):
            beta = self.betas[k]
            gamma = self.gammas[k]
            circuit = self.apply_risk_return(circuit,gamma)
            circuit = self.apply_transaction_cost(circuit,gamma)
            circuit = self.apply_soft_constraint(circuit,gamma)
            circuit = self.apply_QAOA_mixing_operator(circuit,beta)

        circuit = self.apply_measurements(circuit)


        return circuit

    def AOA_circuit(self,D,p=1):
        '''

        Generate the circuir for the alternating ansatz operator

        '''

        # Instantiate the circuit with the symbolic parameters that we will need
        self.gammas = sympy.symbols(["gamma_" + str(k) for k in range(p)])
        self.betas = sympy.symbols(["beta_" + str(k) for k in range(p)])

        circuit = cirq.Circuit()
        circuit = self.prepare_AOA_initial_state(circuit, D=D)

        for k in range(0, p):
            beta = self.betas[k]
            gamma = self.gammas[k]
            circuit = self.apply_risk_return(circuit, gamma)
            circuit = self.apply_transaction_cost(circuit, gamma)
            circuit = self.apply_AOA_mixing_operator(circuit, beta)

        circuit = self.apply_measurements(circuit)

        return circuit

    def exp_ZZ(self,angle,qubit1,qubit2):
        '''

        returns Exp[i*angle*ZZ]

        '''

        return cirq.ZZPowGate(exponent= (angle/sympy.pi)).on(qubit1,qubit2)

    def exp_XX_YY(self,angle,qubit1,qubit2):
        '''

        returns Exp[i*angle*(XX+YY)]

        '''


        return cirq.ISwapPowGate(exponent= (4.0*angle/sympy.pi)).on(qubit1,qubit2)


        return None

    def exp_X(self,angle,qubit):
        '''

        returns Exp[i*angle*X]

        '''
        return cirq.rx(-2*angle).on(qubit)

    def exp_Z(self,angle,qubit):
        '''

        returns Exp[i*angle*Z]

        '''

        return cirq.rz(-2 * angle).on(qubit)

    def prepare_AOA_initial_state(self,circuit,D):
        '''
        Prepares the alternating Operator Ansatz
        initial state.

        |psi > = |10>^D \otimes ( (1/sqrt[2])*|00>+(1/sqrt[2])*|11>)^{N-D}

        Note that our convention is: |x^+ x^- > which is opposite to the paper
        '''

        # Prepare the |10> states, D-times
        for i in range(int(D)):
            sp_i, sm_i = self.portfolio_indices[i][1], self.portfolio_indices[i][2]
            circuit.append(cirq.X(self.qubits[sp_i]))

        # Prepare the Bell states ( (1/sqrt[2])*|00>+(1/sqrt[2])*|11>)^{N-D}
        for i in range(int(D), self.N_portfolio):
            sp_i, sm_i = self.portfolio_indices[i][1], self.portfolio_indices[i][2]
            circuit.append(cirq.H(self.qubits[sp_i]))
            circuit.append(cirq.CNOT(self.qubits[sp_i], self.qubits[sm_i]))

        return circuit


    def apply_hadamard(self,circuit):
        '''
        Applies a Hadamard gate to all quibits
        '''

        hadamard_operators = [cirq.H(self.qubits[i]) for i in range(len(self.qubits))]
        circuit.append(hadamard_operators)

        return circuit

    def apply_QAOA_mixing_operator(self,circuit,beta):
        '''
        Applies the QAOA mixing operator to the circuit

        U(beta) = Exp[-i*beta * \sum_i X_i]

        '''

        #mixer_operators = [cirq.rx(2 * beta).on(self.qubits[i]) for i in range(self.N_qubits)]

        mixer_operators = [self.exp_X(-beta,self.qubits[i]) for i in range(self.N_qubits)]

        circuit.append(mixer_operators)

        return circuit

    def apply_AOA_mixing_operator(self,circuit,beta):
        '''
        Apply the Quantum Alternating-Ansatz mixing operator

        B_odd/even =  X_a X_{a+1} + Y_a Y_{a+1}

        U(B) = Exp[-i beta B]

        As in the ArXiv: 1911.05296, we user long and short position parity mixers

        '''

        # Short position parity mixer
        # Over the X_odd X_{odd+1}+Y_odd Y_{odd+1}
        for i in range(1,self.N_qubits-2,2):
            circuit.append(self.exp_XX_YY(angle=-beta,qubit1=self.qubits[i],qubit2=self.qubits[i+2]))

        # Long position parity mixer
        # Over the X_even X_{even+1}+Y_even Y_{even+1}
        for i in range(0,self.N_qubits-2,2):
            circuit.append(self.exp_XX_YY(angle=-beta,qubit1=self.qubits[i],qubit2=self.qubits[i+2]))

        # Final short position mixer
        circuit.append(self.exp_XX_YY(angle=-beta,qubit1=self.qubits[self.N_qubits-1],qubit2=self.qubits[1]))

        # Final long position mixer
        circuit.append(self.exp_XX_YY(angle=-beta,qubit1=self.qubits[self.N_qubits - 2], qubit2=self.qubits[0]))

        return circuit

    def apply_measurements(self,circuit,key='m'):
        '''
        Applies the measurement operator all qubits on the circuit. The
        default key for the measurement is 'm'
        '''

        measurements = cirq.measure(*self.qubits, key=key)

        circuit.append(measurements)

        return circuit


    def apply_transaction_cost(self,circuit,gamma):
        '''
        Apply the transaction cost model

        C_{TC}(Z) = \sum_i (1-\delta(Zi-yi))*T

        T = transaction cost
        \delta(x-y) = { 1 if x=y, otherwise 0}

        '''

        for i in range(self.N_portfolio):

            sp_i, sm_i = self.portfolio_indices[i][1], self.portfolio_indices[i][2]

            angle1 =  -(1/4)*self.T*(1-self.y[i]**2-self.y[i])*gamma
            circuit.append(self.exp_Z(angle1,self.qubits[sp_i]))

            angle2 = -(1/4)*self.T*(1-self.y[i]**2+self.y[i])*gamma
            circuit.append(self.exp_Z(angle2,self.qubits[sm_i]))

            angle3 = -(1/4)*self.T*(2*self.y[i]**2-1)*gamma
            circuit.append(self.exp_ZZ(-2*angle3,self.qubits[sp_i],self.qubits[sm_i]))


        return circuit


    def apply_risk_return(self,circuit,gamma):
        '''

        Applies the circuit that represents the Portfolio risk-return
        cost function
        '''


        for i in range(self.N_portfolio):

            sp_i, sm_i = self.portfolio_indices[i][1],self.portfolio_indices[i][2]

            # Exp[-i*\sum_i mu_i/2(s^+_i -s^-_i)]
            circuit.append(self.exp_Z((1-self.lam)*gamma*self.mu[i]/2,self.qubits[sp_i] ))
            circuit.append(self.exp_Z(-(1-self.lam)*gamma*self.mu[i]/2,self.qubits[sm_i] ))

            for j in range(self.N_portfolio):

                sp_j, sm_j = self.portfolio_indices[j][1],self.portfolio_indices[j][2]

                if(i !=j):
                    circuit.append(self.exp_ZZ(-self.lam*gamma*(self.sigma[i][j]/4),self.qubits[sp_i],self.qubits[sp_j]))
                    circuit.append(self.exp_ZZ(self.lam*gamma *(self.sigma[i][j]/4), self.qubits[sp_i], self.qubits[sm_j]))
                    circuit.append(self.exp_ZZ(self.lam*gamma *(self.sigma[i][j]/4), self.qubits[sm_i], self.qubits[sp_j]))
                    circuit.append(self.exp_ZZ(-self.lam*gamma *(self.sigma[i][j]/4), self.qubits[sm_i], self.qubits[sm_j]))

                else:
                    circuit.append(self.exp_ZZ(self.lam*gamma *(self.sigma[i][j]/4), self.qubits[sp_i], self.qubits[sm_j]))
                    circuit.append(self.exp_ZZ(self.lam*gamma *(self.sigma[i][j]/4), self.qubits[sm_i], self.qubits[sp_j]))

        return circuit


    def apply_soft_constraint(self,circuit,gamma):
        '''
        Constructs the circuit that represents the soft constraint and applies
        the Gamma angle.

        C = A*(\sum_i Z_i -D)^2

        U = Exp[-i gamma * C]
        '''

        self.A = sympy.symbols("A")

        for i in range(self.N_portfolio):

            sp_i, sm_i = self.portfolio_indices[i][1],self.portfolio_indices[i][2]

            # Exp[-i*AD*\sum_i (s^+_i -s^-_i)]
            circuit.append(self.exp_Z(-gamma*self.A*self.D,self.qubits[sp_i] ))
            circuit.append(self.exp_Z(gamma*self.A*self.D,self.qubits[sm_i] ))

            for j in range(self.N_portfolio):

                sp_j, sm_j = self.portfolio_indices[j][1],self.portfolio_indices[j][2]


                if(i !=j):
                    circuit.append(self.exp_ZZ(-gamma*(self.A/4),self.qubits[sp_i],self.qubits[sp_j]))
                    circuit.append(self.exp_ZZ(gamma * (self.A / 4), self.qubits[sp_i], self.qubits[sm_j]))
                    circuit.append(self.exp_ZZ(gamma * (self.A / 4), self.qubits[sm_i], self.qubits[sp_j]))
                    circuit.append(self.exp_ZZ(-gamma * (self.A / 4), self.qubits[sm_i], self.qubits[sm_j]))

                else:
                    circuit.append(self.exp_ZZ(gamma * (self.A / 4), self.qubits[sp_i], self.qubits[sm_j]))
                    circuit.append(self.exp_ZZ(gamma * (self.A / 4), self.qubits[sm_i], self.qubits[sp_j]))

        return circuit

    def measure_circuit(self,circuit,parameters={}, betas= None,gammas=None,key='m',n_trials=100):
        '''
        This function resolves the input parameters and carries out the measurements of the circuits.
        All symbolic parameters are resolved into values for calculation.

        Keyword arguments:
        A - the penalty scaling coefficient
        D - the investment constraint
        T - the normalized cost incurred if an asset is traded
        mu -  the normalized average asset returns vector
        sigma - the normalized asset returns covariance matrix
        y - the previous portfolio position
        lam - the asset manager control parameter
        betas - the vector representing the angles for the mixer
        gammas - the vector representing the angles for the cost function unitarity matrix
        key - the key that is used for specifying what to measure in the circuit simulation.
        n_trials - the number of times to run the circuit and collect the results
        '''

        # Retrieve the input parameters
        lam = parameters['lam']
        A = parameters['A']
        D = parameters['D']
        mu = parameters['mu']
        sigma = parameters['sigma']
        T = parameters['T']
        y = parameters['y']


        resolved_params ={}

        # The cirq simulator object
        simulator = cirq.Simulator()

        # This parameters only exists for the QAOA cicuit with soft constraints
        try:
            if(A is not None):
                resolved_params[self.A] = A
        except:
            pass

        if(D is not None):
            resolved_params[self.D] = D

        if(lam is not None):
            resolved_params[self.lam] = lam

        if(T is not None):
            resolved_params[self.T] = T

        if(y is not None):

            for k in range(self.N_portfolio):
                resolved_params[self.y[k]] = y[k]

        if(mu is not None):

            for k in range(self.N_portfolio):
                resolved_params[self.mu[k]] = mu[k]

        if(betas is not None):

            for k in range(len(betas)):
                resolved_params[self.betas[k]] = betas[k]

        if(gammas is not None):

            for k in range(len(gammas)):
                resolved_params[self.gammas[k]] = gammas[k]

        if(sigma is not None):
            for k1 in range(self.N_portfolio):
                for k2 in range(self.N_portfolio):
                    resolved_params[self.sigma[k1][k2]] = sigma[k1,k2]

        # Resolve all of the symbolic parameters in the circuit
        resolver = cirq.ParamResolver(resolved_params)

        # Carry out the n_trial measurements
        results = simulator.run(circuit,resolver,repetitions=n_trials)

        # Extract the bitstrings that were measured
        bitstrings = results.measurements[key]

        return bitstrings

    def convert_bitstring_to_z(self,x):
        '''
        Convert a bit string of the measured qubits to
        portfolio holding.

        f(x+,x-) = x+ - x- = y

        example:
        x=[1,0,0,0] -> y=[1,0]
        '''

        x = np.array(x).astype(int)
        z = np.zeros(self.N_portfolio).astype(int)

        for i in range(self.N_portfolio):
            sp_i, sm_i  = self.portfolio_indices[i][1],self.portfolio_indices[i][2]
            z[i] =  x[sp_i]-x[sm_i]

        return z

    def compute_penalty(self,A,D,zi):
        '''

        Compute the soft-constraint Penalty function for 

        penalty = A*(\sum Z_i -D)^2

        keyword arguments:
        A - The penalty scaling parameter
        D - The investment constraint
        '''

        penalty = A*(np.sum(zi)-D)**2

        return penalty

    def compute_portfolio_cost(self,lam,mu,sigma,zi):
        '''

        The portfolio cost function for a single bitstring representing the
        holdings.

        C_RR(z) = \lambda \sum_{i,j} \sigma_{i,j} z_i z_j -(1-\lambda)*\sum_i \mu_i z_i

        keyword arguments:
        lam - Asset manager control parameter
        mu -  The normalized average asset returns vector
        sigma - The normalized asset returns covariance matrix
        '''

        # The cost stemming from the risk
        c_sigma = 0.0
        for k1 in range(self.N_portfolio):
            for k2 in range(self.N_portfolio):
                c_sigma += sigma[k1,k2]*zi[k1]*zi[k2]

        # the cost stemming from the returns
        c_mu = 0.0
        for k1 in range(self.N_portfolio):
            c_mu+=mu[k1]*zi[k1]

        cost = lam*c_sigma-1.0*(1.0-lam)*c_mu

        return cost

    def delta(self,a,b):
        '''
        Compute the Kroneker delta function for arguments a,b
        '''

        x = 0.0

        if(a==b):
            x = 1.0

        return x

    def compute_portfolio_volatility(self,sigma,zi):
        '''
        volatility^2 = \sum_{ij}\sigma_{ij} z_i z_j
        '''

        c_sigma = 0.0
        for k1 in range(self.N_portfolio):
            for k2 in range(self.N_portfolio):
                c_sigma += sigma[k1,k2]*zi[k1]*zi[k2]

        return np.sqrt(c_sigma)

    def compute_portfolio_returns(self,mu,zi):
        '''
        portfolio_returns = \sum_i \mu_i z_i
        '''

        c_mu = 0.0
        for k1 in range(self.N_portfolio):
            c_mu += mu[k1] * zi[k1]

        return c_mu

    def compute_transaction_cost(self,T,y,zi):
        '''

        Transaction costs
        C_T = \sum_i T*(1-\delta(y_i-z_i))

        keyword arguments:
        T - transactions costs
        y - previous portfolio holdings
        z - the portfolio holdings vector
        '''

        t_cost = 0.0

        for k1 in range(self.N_portfolio):
            t_cost+= T*(1.0-self.delta(zi[k1],y[k1]))

        return t_cost

    def compute_transaction_cost_expectation_value(self,parameters,portfolio_holdings):
        '''

        The expectation value of the cost function

        keyword arguments:
        portfolio_holdings - dictionary of all the simulated portfolio holdings
        T - Transaction costs
        y - vector with previous portfolio holdings
        '''

        expectation_value = 0

        T = parameters['T']
        y = parameters['y']

        for i in range(len(portfolio_holdings['labels'])):
            zi = portfolio_holdings['state_vector'][i]
            prob_i = portfolio_holdings['probability'][i]
            expectation_value+= prob_i*self.compute_transaction_cost(T,y,zi)

        return expectation_value

    def compute_portfolio_cost_expectation_value(self,parameters,portfolio_holdings):
        '''


        keyword arguments:
        portfolio_holdings -
        sigma -
        mu -
        lam -
        '''

        expectation_value = 0

        lam = parameters['lam']
        mu = parameters['mu']
        sigma = parameters['sigma']

        for i in range(len(portfolio_holdings['labels'])):
            zi = portfolio_holdings['state_vector'][i]
            prob_i = portfolio_holdings['probability'][i]
            expectation_value+= prob_i*self.compute_portfolio_cost(lam,mu,sigma,zi)

        return expectation_value


    def compute_penalty_expectation_value(self,parameters,portfolio_holdings):
        '''

        Compute the cost of the soft constraint

        '''

        A = parameters['A']
        D = parameters['D']


        expectation_value = 0

        for i in range(len(portfolio_holdings['labels'])):
            zi = portfolio_holdings['state_vector'][i]
            prob_i = portfolio_holdings['probability'][i]
            expectation_value+= prob_i*self.compute_penalty(A,D,zi)

        return expectation_value

    def compute_total_cost_expectation_value(self, portfolio_holdings,parameters):
        '''
        Compute the expected value of a state
        '''

        expectation_value = 0

        A = parameters['A']
        D = parameters['D']
        lam = parameters['lam']
        mu = parameters['mu']
        sigma = parameters['sigma']
        T = parameters['T']
        y = parameters['y']


        for i in range(len(portfolio_holdings['labels'])):
            zi = portfolio_holdings['state_vector'][i]
            prob_i = portfolio_holdings['probability'][i]
            portfolio_cost = self.compute_portfolio_cost(lam, mu, sigma, zi)
            penalty_cost = self.compute_penalty(A,D,zi)
            transaction_cost = self.compute_transaction_cost(T,y,zi)
            total_cost = portfolio_cost+penalty_cost+transaction_cost
            expectation_value += prob_i * total_cost

        return expectation_value


    def circuit_measurement_function(self,x,circuit,parameters,n_trials=100,p=1):
        '''
        This function optimizes the circuit
        '''

        gammas = x[0:p]
        betas = x[p:]
        bitstrings = self.measure_circuit(circuit, parameters=parameters, betas=betas, gammas=gammas, n_trials=n_trials)
        portfolio_holdings = self.convert_bitstrings_to_portfolio_holdings(bitstrings)
        energy_expectation_value = self.compute_total_cost_expectation_value(portfolio_holdings, parameters)

        return energy_expectation_value

    def optimize_circuit_GD(self,circuit,parameters,n_trials,p,lr=0.2,steps=10):
        '''
        Gradient descent optimization
        '''


        # Initialize random gradients
        E_gradients = np.random.rand(2*p)
        x = np.random.rand(2*p)
        dx = np.zeros(2 * p)

        for k in tqdm(range(steps)):

           # Compute the energy
            #E = self.circuit_measurement_function(x,circuit,parameters,n_trials=n_trials,p=p)

            for i in range(len(x)):
                dx[:] = 0.0
                dx[i] = lr * E_gradients[i]
                E_pdx = self.circuit_measurement_function(x + dx, circuit, parameters, n_trials=n_trials, p=p)
                E_mdx = self.circuit_measurement_function(x - dx, circuit, parameters, n_trials=n_trials, p=p)
                E_gradients[i] = 0.5*((E_pdx-E_mdx)/np.sqrt(np.sum(dx**2)))

            #update the parameters
            x = x-lr*E_gradients
            #x = np.array([x[i]+2.0*np.pi if x[i] < 0.0 else x[i]-2.0*np.pi if x[i] > 2.0*np.pi else x[i] for i in range(len(x))])

        # get the other results
        gammas = x[0:p]
        betas = x[p:]
        bitstrings = self.measure_circuit(circuit, parameters=parameters, betas=betas, gammas=gammas, n_trials=n_trials)
        portfolio_holdings = self.convert_bitstrings_to_portfolio_holdings(bitstrings)
        energy_expectation_value = self.compute_total_cost_expectation_value(portfolio_holdings, parameters)
        best_solutions = self.determine_best_solution_from_trials(parameters, portfolio_holdings)

        results={}
        results['portfolio_holdings'] = portfolio_holdings
        results['best_solutions'] = best_solutions
        results['optimal_gammas'] = gammas
        results['optimal_betas'] =  betas
        results['optimal_energy_measurement'] = energy_expectation_value

        return results


    def optimize_circuit(self,circuit,parameters,n_trials,p,maxiter=50):
        '''
        Carry out the optimization of a specified circuit using the scipy-minimizer

        '''

        x0 = np.random.rand(2*p)
        res = minimize(self.circuit_measurement_function, x0,
                       args =(circuit,parameters,n_trials,p),
                       method='nelder-mead',
                       options = {'maxiter': maxiter, 'disp': True})

        print('=' * 100)
        print('optimal cost: ', res.fun)
        print('='*100)

        # get the other results
        gammas = res.x[0:p]
        betas = res.x[p:]
        bitstrings = self.measure_circuit(circuit, parameters=parameters, betas=betas, gammas=gammas, n_trials=n_trials)
        portfolio_holdings = self.convert_bitstrings_to_portfolio_holdings(bitstrings)
        energy_expectation_value = self.compute_total_cost_expectation_value(portfolio_holdings, parameters)


        best_solutions = self.determine_best_solution_from_trials(parameters,portfolio_holdings)

        results={}
        results['portfolio_holdings'] = portfolio_holdings
        results['best_solutions'] = best_solutions
        results['optimal_gammas'] = gammas
        results['optimal_betas'] =  betas
        results['optimal_energy_measurement'] = energy_expectation_value

        return results

    def optimize_circuit_angles_cross_entropy(self,circuit,parameters,p,n_trials,iterations,f_elite,Nce_samples):
        '''
        This function uses the cross-entropy method to optimize the circuit angle parameters

        keyword arguments:
        parameters -
        p -
        n_trials -
        iterations -
        f_elite -
        Nce_samples -
        '''

        # The intitial values of the returns and the parameter covariance
        sigma = np.identity(2*p)
        mu = np.random.uniform(0.0,2*np.pi,2*p)

        for i in range(iterations):
            # Generate N-samples from the multivariate Gaussian
            X = np.random.multivariate_normal(mu,sigma,Nce_samples)

            # we use a truncated Gaussian, so we filter values accordilngly
            # filter these samples so all angles lie between 0 and 2*pi
            # X = X[ [ all([(X[i,k]>= 0.0) & (X[i,k]<= 2.0*np.pi) for k in range(len(X[0]))]) for i in range(len(X))]]
            #
            # # keep sampling until number of required samples is reached
            # k_samples = len(X)
            # while(k_samples <= Nce_samples):
            #     X_sample = np.random.multivariate_normal(mu,sigma,Nce_samples)
            #     X_filtered = X_sample[ [ all([(X_sample[i,k]>= 0.0) & (X_sample[i,k]<= 2.0*np.pi) for k in range(len(X_sample[0]))]) for i in range(len(X_sample))]]
            #     X = np.concatenate((X,X_filtered))
            #     k_samples = len(X)

            # Remove the remaining samples
            #X = X[0:Nce_samples]

            # check that

            E = np.zeros(len(X))
            data = []

            for k in tqdm(range(len(X))):
                E[k] = self.circuit_measurement_function(x=X[k], circuit=circuit, parameters=parameters, n_trials=n_trials, p=p)
                data.append([*X[k],E[k]])

            # Sort the value according to the best
            sorted_data = np.array(sorted(data, key=lambda x: x[2*p], reverse=False))

            #print(sorted_data)

            # Now compute the new averages
            N_elite = int(f_elite*Nce_samples)
            X_elite = sorted_data[0:N_elite,0:2*p]
            E_elite = sorted_data[0:N_elite,2*p:]

            print('k={}, mean_E={:.6f}, std_E={:.6f}, gammas[pi]={}, betas[pi]={}'.format(i,E_elite.mean(),E_elite.std(),  mu[0:p]/np.pi,  mu[p:]/np.pi))

            # Compute the new values of mu and sigma
            mu = np.mean(X_elite,axis=0)
            sigma = np.cov(X_elite.T)


        # get the other results
        gammas = mu[0:p]
        betas = mu[p:]
        bitstrings = self.measure_circuit(circuit, parameters=parameters, betas=betas, gammas=gammas, n_trials=n_trials)
        portfolio_holdings = self.convert_bitstrings_to_portfolio_holdings(bitstrings)
        energy_expectation_value = self.compute_total_cost_expectation_value(portfolio_holdings, parameters)

        best_solutions = self.determine_best_solution_from_trials(parameters, portfolio_holdings)

        results = {}
        results['portfolio_holdings'] = portfolio_holdings
        results['best_solutions'] = best_solutions
        results['optimal_gammas'] = gammas
        results['optimal_betas'] = betas
        results['optimal_energy_measurement'] = energy_expectation_value


        return results


    def count_instances(self, bitstrings):
        '''
        Counts the number of instances for a collection of bistring samples
        '''

        # The dictionary to store the results
        results = {}

        # The values that can be taken by Zi
        index = [0, 1]

        keys = itertools.product(index, repeat=self.N_qubits)

        labels = []
        label_count = []

        for state in list(keys):

            # Convert the state to integer array
            Z_state = np.array([state[i] for i in range(len(state))]).astype(int)

            # label the state
            state_label = ''.join(Z_state.astype('str'))

            state_count = 0

            for i in range(len(bitstrings)):

                # retrieve the ith-bitstring
                xi = bitstrings[i]

                if (np.array_equal(Z_state, xi) == True):
                    state_count += 1

            labels.append(state_label)
            label_count.append(state_count)

        results['labels'] = labels
        results['counts'] = label_count
        results['probability'] = np.array(label_count) / np.sum(np.array(label_count))
        return results

    def convert_bitstrings_to_portfolio_holdings(self,bitstrings):
        '''

        Convert the bitstring from the individual qubit measurements to
        the corresponding values that correspond to portfolio positions:
        ie. Zi = x^+_i - x^-_i
        '''



        results = {}

        index = [-1,0,1]

        keys = itertools.product(index,repeat=self.N_portfolio)

        labels = []
        label_count = []
        state_vector = []

        for state in list(keys):

            # Convert the state to integer array
            Z_state = np.array([state[i] for i in range(len(state))]).astype(int)

            # label the state
            state_label = ''.join(Z_state.astype('str'))

            state_count = 0

            for i in range(len(bitstrings)):

                # retrieve the ith-bitstring
                xi = bitstrings[i]
                zi = self.convert_bitstring_to_z(xi)

                if (np.array_equal(Z_state, zi) == True):
                    state_count += 1

            labels.append(state_label)
            label_count.append(state_count)
            state_vector.append(Z_state)

        results['labels'] = labels
        results['counts'] = label_count
        results['probability'] = np.array(label_count) / np.sum(np.array(label_count))
        results['state_vector'] = np.array(state_vector)

        return results

    def determine_best_solution_from_trials(self,parameters,portfolio_holdings):
        '''
        For a set of results from measurements, determine the best solutions
        '''

        lam = parameters['lam']
        A = parameters['A']
        D = parameters['D']
        mu = parameters['mu']
        sigma = parameters['sigma']
        T = parameters['T']
        y = parameters['y']

        prob = np.array(portfolio_holdings['probability'])

        nonzero_indx = (prob>0)

        portfolio_holdings_states_np = np.array(portfolio_holdings['state_vector'])
        portfolio_holdings_probability_np = np.array(portfolio_holdings['probability'])

        nonzero_states = portfolio_holdings_states_np[nonzero_indx]
        nonzero_probabilities = portfolio_holdings_probability_np[nonzero_indx]

        # Initialize the energy
        energies = np.ones(len(nonzero_states))

        # Determine the energy of these states and find the best solutions among the subset
        k=0
        for zi in nonzero_states:
            E_penalty = self.compute_penalty(A,D,zi)
            E_portfolio  = self.compute_portfolio_cost(lam,mu,sigma,zi)
            E_transaction_cost =self.compute_transaction_cost(T,y,zi)
            E_zi = E_penalty+E_portfolio+E_transaction_cost
            energies[k] = E_zi
            k+=1

        energies = np.array(energies)

        # Get the minimum energy of these experiments
        E_min = min(energies)
        E_min_indx = np.argwhere(energies == E_min).flatten()

        min_states = nonzero_states[E_min_indx]
        min_states_probabilities = nonzero_probabilities[E_min_indx]

        results={}
        results['minimum_cost'] = E_min
        results['minimum_cost_states'] = min_states
        results['minimum_cost_probabilities'] = min_states_probabilities

        return results

    def grid_search(self,circuit,parameters,N_grid,n_trials=100):
        '''
        Carry out a grid search of a circuit at depth p=1
        '''

        # Retrieve the parameters
        #lam = parameters['lam']
        #A = parameters['A']
        #D = parameters['D']
        #mu = parameters['mu']
        #sigma = parameters['sigma']
        #T = parameters['T']
        #y = parameters['y']

        results = {}

        total_cost_grid = np.zeros((N_grid,N_grid))
        penalty_cost_grid = np.zeros((N_grid, N_grid))
        portfolio_cost_grid = np.zeros((N_grid, N_grid))
        transaction_cost_grid = np.zeros((N_grid, N_grid))

        betas = np.linspace(0.0,2.0*np.pi,N_grid)
        gammas = np.linspace(0.0,2.0*np.pi,N_grid)

        min_cost = 1e10
        min_gamma = 0.0
        min_beta = 0.0
        min_holdings= None

        for k1 in range(len(betas)):
            for k2 in range(len(gammas)):
                beta = betas[k1]
                gamma = gammas[k2]
                bitstrings = self.measure_circuit(circuit,parameters=parameters,betas=[beta],gammas=[gamma],n_trials=n_trials)
                portfolio_holdings = self.convert_bitstrings_to_portfolio_holdings(bitstrings)

                penalty_cost = self.compute_penalty_expectation_value(parameters,portfolio_holdings)
                transaction_cost = self.compute_transaction_cost_expectation_value(parameters,portfolio_holdings)
                portfolio_cost = self.compute_portfolio_cost_expectation_value(parameters,portfolio_holdings)
                total_cost = penalty_cost+transaction_cost+portfolio_cost

                transaction_cost_grid[k1,k2] = transaction_cost
                penalty_cost_grid[k1,k2] = penalty_cost
                portfolio_cost_grid[k1,k2] = portfolio_cost
                total_cost_grid[k1,k2] = total_cost

                if (total_cost < min_cost):
                    min_cost = total_cost
                    min_gamma = gamma
                    min_beta = beta
                    min_holdings = portfolio_holdings



        results['minimum_cost'] = min_cost
        results['min_gamma'] = min_gamma
        results['min_betas'] = min_beta
        results['min_portfolio_holdings'] = min_holdings
        results['parameters'] = parameters

        results['beta_grid'] = betas
        results['gamma_grid'] = gammas


        results['total_cost_grid'] = total_cost_grid
        results['portfolio_cost_grid'] = portfolio_cost_grid
        results['penalty_cost_grid'] = penalty_cost_grid
        results['transaction_cost_grid'] = transaction_cost_grid

        return results


    def plot(self,parameters):

        optimizer_name = parameters['optimizer_name']
        optimizer_data = parameters['optimizer_data']
        p_depth = parameters['p_depth']
        figure_number =parameters['figure_number']

        fig = plt.figure()
        fig.set_figheight(10)
        fig.set_figwidth(18)
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(212)

        ax1.set_title(r'$\langle \psi| C(z)|\psi \rangle$ vs $p$ [{}]'.format(optimizer_name) , fontsize=25)
        ax1.set_xlabel(r"$p$-depth", fontsize=15)
        ax1.set_ylabel(r"$\langle \psi| C(z)|\psi \rangle$", fontsize=15)
        ax1.plot(p_depth, [optimizer_data[k]['optimal_energy_measurement'] for k in range(len(p_depth))]
                 , marker='o',
                 markersize=15)

        ax2.set_title(r'$\langle x*| C(z)|x* \rangle$ vs $p$ [{}]'.format(optimizer_name), fontsize=25)
        ax2.set_ylabel(r"$\langle x*| C(z)|x* \rangle$", fontsize=15)
        ax2.set_xlabel(r"$p$-depth", fontsize=15)
        ax2.plot(p_depth,
                 [optimizer_data[k]['best_solutions']['minimum_cost'] for k in range(len(p_depth))],
                 marker='o',
                 markersize=15)

        ax3.set_title(r'$p(x*)$ vs $p$ [{}]'.format(optimizer_name), fontsize=25)
        ax3.set_xlabel(r"$p$-depth", fontsize=15)
        ax3.set_ylabel(r"$p(x*)$", fontsize=15)
        ax3.plot(p_depth, [optimizer_data[k]['best_solutions']['minimum_cost_probabilities'] for k in
                           range(len(p_depth))],
                 marker='o',
                 markersize=15)
        fig.tight_layout()

        text = r'''
        Figure {}: The convergence of the expectation value $\langle \psi|C(z)|\psi \rangle$ (upper left), the expectation value of 
        the lowest energy solution $|x*\rangle$ (upper right), and the associated probability of the lowest energy  
        state $p(x*)$ (bottom) as a function of the circuit depth $p$. The {} optimization routine was used to 
        obtain the results.
        '''.format(figure_number,optimizer_name)
        fig.text(.1, -.17, text, ha='left', size=20)

        return fig

    def json_default(self,obj):

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError('Not serializable')

    def save_data_as_json(self,results,f_name):
        '''
        Save the results of

        '''

        with open(f_name,'w') as fl:
            json.dump(results,fl,default=self.json_default)

        return None

    def load_data_from_json(self,f_name):

        with open(f_name) as json_file:
            json_load = json.load(json_file)

        return json_load