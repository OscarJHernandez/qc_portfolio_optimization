import cirq
import sympy
import numpy as np
import itertools
import matplotlib.pyplot as plt



class Portfolio():

    def __init__(self, N_portfolio=0, mu=None, sigma=None,lam=None):
        '''

        '''

        self.N_portfolio = N_portfolio

        self.N_qubits = 2*N_portfolio

        # Initialize the quibits that will be used to represent the portfolio
        self.qubits = [cirq.LineQubit(i) for i in range(self.N_qubits)]

        # The indices required for this problem
        # (portfolio index, s^+_i index, s^-_i index
        self.portfolio_indices = [(i,2*i,2*i+1) for i in range(N_portfolio)]

        # Instantiate the circuit object
        #self.circuit = cirq.Circuit()

        # Instantiate the returns vector
        self.mu = sympy.symbols(["mu_"+str(k) for k in range(N_portfolio)])

        # instantiate the sigma matrix
        self.sigma = sympy.symbols([["sigma_"+str(i)+str(j) for i in range(N_portfolio)] for j in range(N_portfolio)])

        # instantiate the Lambda value
        self.lam = sympy.symbols("L")

        # Instantiate the transaction cost
        self.T = sympy.symbols("T")

        self.D = sympy.symbols("D")

        # instantiate the previous weights
        self.y = sympy.symbols(["y_"+str(k) for k in range(N_portfolio)])

        return None

    def QAOA_circuit(self,p=1):
        '''
        Construct the QAOA circuit with soft constraints
        '''

        circuit = None


        return circuit

    def AOA_circuit(self,p=1):

        circuit = None

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


    def apply_portfolio_lagrangian(self,circuit,gamma):
        '''

        Applies the circuit that represents the Portfolio Lagrangian

        Constructs the circuit that represents the soft constraint

        P_
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

        C = A(\sum_i Z_i -D)^2
        '''

        self.A = sympy.symbols("A")

        for i in range(self.N_portfolio):

            sp_i, sm_i = self.portfolio_indices[i][1],self.portfolio_indices[i][2]

            #circuit.append(cirq.rz(2*gamma*self.A*self.D).on(self.qubits[sp_i]))
            #circuit.append(cirq.rz(-2*gamma*self.A*self.D).on(self.qubits[sm_i]))

            # Exp[-i*AD*\sum_i (s^+_i -s^-_i)]
            circuit.append(self.exp_Z(-gamma*self.A*self.D,self.qubits[sp_i] ))
            circuit.append(self.exp_Z(gamma*self.A*self.D,self.qubits[sm_i] ))

            for j in range(self.N_portfolio):

                sp_j, sm_j = self.portfolio_indices[j][1],self.portfolio_indices[j][2]

                #angle = (-1)*(-2*gamma/sympy.pi) * (self.A / 4)

                if(i !=j):
                    circuit.append(self.exp_ZZ(-gamma*(self.A/4),self.qubits[sp_i],self.qubits[sp_j]))
                    circuit.append(self.exp_ZZ(gamma * (self.A / 4), self.qubits[sp_i], self.qubits[sm_j]))
                    circuit.append(self.exp_ZZ(gamma * (self.A / 4), self.qubits[sm_i], self.qubits[sp_j]))
                    circuit.append(self.exp_ZZ(-gamma * (self.A / 4), self.qubits[sm_i], self.qubits[sm_j]))

                    #circuit.append(cirq.ZZPowGate(exponent= angle).on(self.qubits[sp_i],self.qubits[sp_j]))
                    #circuit.append(cirq.ZZPowGate(exponent= -angle).on(self.qubits[sp_i],self.qubits[sm_j]))
                    #circuit.append(cirq.ZZPowGate(exponent= -angle).on(self.qubits[sm_i],self.qubits[sp_j]))
                    #circuit.append(cirq.ZZPowGate(exponent= angle).on(self.qubits[sm_i],self.qubits[sm_j]))
                else:
                    circuit.append(self.exp_ZZ(gamma * (self.A / 4), self.qubits[sp_i], self.qubits[sm_j]))
                    circuit.append(self.exp_ZZ(gamma * (self.A / 4), self.qubits[sm_i], self.qubits[sp_j]))
                    #circuit.append(cirq.ZZPowGate(exponent= -angle).on(self.qubits[sp_i],self.qubits[sm_j]))
                    #circuit.append(cirq.ZZPowGate(exponent= -angle).on(self.qubits[sm_i],self.qubits[sp_j]))

        return circuit

    def measure_circuit(self,circuit,A=None,D=None,T=None,mu=None,sigma=None,y=None, lam= None,key='m',n_trials=100):
        '''
        This function resolves the input parameters and carries out the measurements.

        Keyword arguments:
        A -
        D -
        T -
        mu -
        sigma -
        y -
        lam -
        key -
        n_trials -
        '''

        resolved_params ={}

        # The cirq simulator object
        simulator = cirq.Simulator()

        if(A is not None):
            resolved_params[self.A] = A

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
        Convert a bit string of the measured qubits

        example:
        [1,0,0,0] -> [1,0]
        '''

        x = np.array(x).astype(int)
        z = np.zeros(self.N_portfolio).astype(int)

        for i in range(self.N_portfolio):
            sp_i, sm_i  = self.portfolio_indices[i][1],self.portfolio_indices[i][2]
            z[i] =  x[sp_i]-x[sm_i]

        return z

    def compute_penalty(self,A,D,zi):
        '''

        Compute the soft-constraint Penalty function

        keyword arguments:
        A -
        D -

        '''

        penalty = A*(np.sum(zi)-D)**2

        return penalty

    def compute_portfolio_cost(self,lam,mu,sigma,zi):
        '''

        The portfolio cost function for a single bitstring representing the
        holdings

        '''

        c_sigma = 0.0
        for k1 in range(self.N_portfolio):
            for k2 in range(self.N_portfolio):
                c_sigma += sigma[k1,k2]*zi[k1]*zi[k2]

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

    def compute_transaction_cost(self,T,y,zi):
        '''

        Transaction costs

        keyword arguments:
        T - transactions costs
        y - previous portfolio holdings
        zi -

        '''

        t_cost = 0.0

        for k1 in range(self.N_portfolio):
            t_cost+= T*(1.0-self.delta(zi[k1],y[k1]))

        return t_cost

    def compute_transaction_cost_expectation_value(self,T,y,portfolio_holdings):
        '''

        The expectation value of the cost function

        keyword arguments:
        portfolio_holdings -
        T - Transaction costs
        y - vector with previous portfolio holdings

        '''

        expectation_value = 0

        for i in range(len(portfolio_holdings['labels'])):
            zi = portfolio_holdings['state_vector'][i]
            prob_i = portfolio_holdings['probability'][i]
            expectation_value+= prob_i*self.compute_transaction_cost(T,y,zi)

        return expectation_value

    def compute_portfolio_cost_expectation_value(self,lam,mu,sigma,portfolio_holdings):
        '''


        keyword arguments:
        portfolio_holdings -
        sigma -
        mu -
        lam -
        '''

        expectation_value = 0

        for i in range(len(portfolio_holdings['labels'])):
            zi = portfolio_holdings['state_vector'][i]
            prob_i = portfolio_holdings['probability'][i]
            expectation_value+= prob_i*self.compute_portfolio_cost(lam,mu,sigma,zi)

        return expectation_value

    def compute_penalty_expectation_value(self,A,D,portfolio_holdings):


        expectation_value = 0

        for i in range(len(portfolio_holdings['labels'])):
            zi = portfolio_holdings['state_vector'][i]
            prob_i = portfolio_holdings['probability'][i]
            expectation_value+= prob_i*self.compute_penalty(A,D,zi)

        return expectation_value

    def count_instances(self, bitstrings):
        '''
        Counts the number of instances
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