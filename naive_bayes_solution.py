from bnetbase import Variable, Factor, BN
import csv
import itertools

def dom_helper(factor,variable):
    restr_sp = [v for v in factor.get_scope() if v != variable]
    rfactor = Factor(f"{factor.name}_restr", restr_sp)
    # possible assignments for restricted_scope
    dmns = [v.domain() for v in restr_sp]
    allass = [[]]

    # all combinations of assignments
    for dmn in dmns:
        nass = []
        for ass in allass:
            for val in dmn:
                nass.append(ass + [val])
        allass = nass
    return rfactor,restr_sp,allass
def normalize(factor):
    '''
    Normalize the factor such that its values sum to 1.
    Do not modify the input factor.

    :param factor: a Factor object.
    :return: a new Factor object resulting from normalizing factor.
    '''
    nml_factor = Factor(factor.name, factor.get_scope())
    ttl = sum(factor.values)
    # print(ttl)
    nml_factor.values = [val / ttl for val in factor.values]
    return nml_factor


def restrict(factor, variable, value):
    '''
    Restrict a factor by assigning value to variable.
    Do not modify the input factor.

    :param factor: a Factor object.
    :param variable: the variable to restrict.
    :param value: the value to restrict the variable to
    :return: a new Factor object resulting from restricting variable to value.
             This new factor no longer has variable in it.
    '''
    rfactor,restr_sp,allass = dom_helper(factor,variable)
    # restricted factor values
    for assignment in allass:
        for i in range(len(restr_sp)):
            restr_sp[i].set_assignment(assignment[i])
        variable.set_assignment(value)
        # print(variable)
        rfactor.add_value_at_current_assignment(factor.get_value_at_current_assignments())
    return rfactor

def sum_out(factor, variable):
    '''
    Sum out a variable from factor.
    Do not modify the input factor.

    :param factor: a Factor object.
    :param variable: the variable to sum out.
    :return: a new Factor object resulting from summing out variable from the factor.
             This new factor no longer has variable in it.
    '''
    sfactor,nsp,allass = dom_helper(factor,variable)
    # Sum out the vara
    for ass in allass:
        for i in range(len(nsp)):
            nsp[i].set_assignment(ass[i])
        #sum over all values of the var
        ttl = 0
        for val in variable.domain():
            variable.set_assignment(val)
            ttl += factor.get_value_at_current_assignments()
        sfactor.add_value_at_current_assignment(ttl)
        # print(sfactor.values)
    return sfactor

def multiply(factor_list):
    '''
    Multiply a list of factors together.
    Do not modify any of the input factors.

    :param factor_list: a list of Factor objects.
    :return: a new Factor object resulting from multiplying all the factors in factor_list.
    '''
    if not factor_list:
        raise ValueError("Factor list is empty.")
    rsc = []
    for factor in factor_list:
        for v in factor.get_scope():
            if v not in rsc:
                rsc.append(v)
                # print(rsc.values)
    rfactor = Factor("Product", rsc)

    # rfc, rfcs, allass = dom_helper(rfactor,None)
    dmns = [v.domain() for v in rsc]
    allass = [[]]
    for dmn in dmns:
        nass = []
        for assignment in allass:
            for val in dmn:
                nass.append(assignment + [val])
        allass = nass
        # print(allass)


    # multip values for each assignment
    for ass in allass:
        for i in range(len(rsc)):
            rsc[i].set_assignment(ass[i])
        pd = 1
        for factor in factor_list:
            pd *= factor.get_value_at_current_assignments()
            # print(pd)
        rfactor.add_value_at_current_assignment(pd)
    return rfactor

# def ve(bayes_net, var_query, EvidenceVars):
#     '''
#     Execute the variable elimination algorithm on the Bayesian network bayes_net
#     to compute a distribution over the values of var_query given the
#     evidence provided by EvidenceVars.
#
#     :param bayes_net: a BN object.
#     :param var_query: the query variable.
#     :param EvidenceVars: the evidence variables.
#     :return: a list of probabilities for each value in the query variable's domain.
#     '''
#     # Stp1 : Restrict
#     for evar in EvidenceVars:
#         facs = []
#         for factor in bayes_net.factors():
#             if evar in factor.get_scope():
#                 facs.append(restrict(factor, evar, evar.get_evidence()))
#             else:
#                 facs.append(factor)
#
#     # Stepr 2: ELiminate by summ out and multiply
#     evars = [v for v in bayes_net.variables() if v != var_query and v not in EvidenceVars]
#     for var in evars:
#         rfacs = [f for f in facs if var in f.get_scope()]
#         facs = [f for f in facs if var not in f.get_scope()]
#         if rfacs:
#             facs.append(sum_out(multiply(rfacs), var))
#     #Step 3: Normalize as needed
#     return normalize(multiply(facs))

def ve(bayes_net, var_query, EvidenceVars):
    '''
    Execute the variable elimination algorithm on the Bayesian network bayes_net
    to compute a distribution over the values of var_query given the
    evidence provided by EvidenceVars.
    :param bayes_net: a BN object.
    :param var_query: the query variable.
    :param EvidenceVars: the evidence variables.
    :return: a list of probabilities for each value in the query variable's domain.
    '''
    # Stp1: Restrict
    facs = bayes_net.factors()
    for evar in EvidenceVars:
        nfacs = []
        for factor in facs:
            if evar in factor.get_scope():
                # print(facs)
                nfacs.append(restrict(factor, evar, evar.get_evidence()))
            else:
                nfacs.append(factor)
        facs = nfacs #DO not remove, causes test to fail

    # Step 2: Eliminate by sum out and multiply
    evars = [v for v in bayes_net.variables() if v != var_query and v not in EvidenceVars]
    for var in evars:
        rfacs = [f for f in facs if var in f.get_scope()]
        facs = [f for f in facs if var not in f.get_scope()]
        if rfacs:
            facs.append(sum_out(multiply(rfacs), var))
            # print(rfacs)
    return normalize(multiply(facs))
def naive_bayes_model(data_file, variable_domains={
    "Work": ['Not Working', 'Government', 'Private', 'Self-emp'],
    "Education": ['<Gr12', 'HS-Graduate', 'Associate', 'Professional', 'Bachelors', 'Masters', 'Doctorate'],
    "Occupation": ['Admin', 'Military', 'Manual Labour', 'Office Labour', 'Service', 'Professional'],
    "MaritalStatus": ['Not-Married', 'Married', 'Separated', 'Widowed'],
    "Relationship": ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'],
    "Race": ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'],
    "Gender": ['Male', 'Female'],
    "Country": ['North-America', 'South-America', 'Europe', 'Asia', 'Middle-East', 'Carribean'],
    "Salary": ['<50K', '>=50K']
}, class_var=Variable("Salary", ['<50K', '>=50K'])):
    '''
    NaiveBayesModel returns a BN that is a Naive Bayes model that
    represents the joint distribution of value assignments to
    variables in the Adult Dataset from UCI.
    @return a BN that is a Naive Bayes model and which represents the Adult Dataset.
    '''
    counts = {}
    cnts = {}
    variables = {}
    factors = []
    input_data = []
    # READ IN THE DATA
    with open(data_file, 'r') as file:
        lines = file.readlines()
        headers = lines[0].strip().split(",")  # extract header row
        # for i in range(len(headers)):
        #         head_ind[headers[i]] = i
        for line in lines[1:]:
            input_data.append(line.strip().split(","))

    # Initialize variables
    for var in variable_domains:
        variables[var] = Variable(var, variable_domains[var])
        counts[var] = {}
        for cls in class_var.domain(): # initializing to clean up code
            counts[var][cls] = {}
            cnts[cls] = 0

    #read the inputdata and form revlevant datas
    for row in input_data:
        rowr = dict(zip(headers, row)) # Use zip to clean up ode
        sal = rowr["Salary"] #using variable to domain rference

        # print(sal)
        cnts[sal] += 1
        for var, value in rowr.items():
            if var != "Salary": 
                if value not in counts[var][sal]:
                    counts[var][sal][value] = 0 #Initialize all values to 0
                    # print(counts)
                counts[var][sal][value] += 1

    # sfactor = Factor("Salary", [variables["Salary"]])
    # tot = 0
    # for cls in class_var.domain():
    #     tot += cnts[cls] # to get tot number of things as needed
    #     prob = cnts[cls] / tot #probablity of releventa
    #     sprob.append([cls, prob])
    # sfactor.add_values(sprob)
    # factors.append(sfactor)

    #factor for salary: Factor1
    sfactor = Factor("Salary", [variables["Salary"]])
    sprob = [cnts[cls] / sum(cnts.values()) for cls in class_var.domain()]
    sfactor.add_values([[cls, prob] for cls, prob in zip(class_var.domain(), sprob)])
    factors.append(sfactor) # Initial salary factor tow ork with
    # print(factors)

    #conditional factors for other variables
    for var, variable in variables.items():
        if var == "Salary":
            continue
        factor = Factor(var, [variable, variables["Salary"]])#Factor2
        #pribabilities for each combination
        for sal in class_var.domain():
            for val in variable_domains[var]:
                prob = (counts[var][sal].get(val, 0) / cnts[sal]
                        if cnts[sal] > 0 else 0) # calcualate abs probabability
                # print(prob)
                factor.add_values([[val, sal, prob]])
        factors.append(factor)
        # print(factors)
    return BN("SN Bayes Model", list(variables.values()), factors)

def explore(bayes_net, question):
    '''
    Analyze fairness questions using the Bayes Net with optimizations.
    Only using allowed imports: from bnetbase import Variable, Factor, BN; import csv; import itertools
    '''
    e1_set = ["Work", "Occupation", "Education", "Relationship"]
    res = {}  # Cache for probability computations

    # Initialize counters dictionary
    ctrs = {
        'f_e1_gt_e2': 0,
        'm_e1_gt_e2': 0,
        'f_e1_gt_half_cr': 0,
        'f_e1_gt_half_tot': 0,
        'm_e1_gt_half_cr': 0,
        'm_e1_gt_half_tot': 0,
        'f_tot': 0,
        'm_tot': 0
    }

    with open('data/adult-test.csv', 'r') as file:
        reader = csv.reader(file)
        headers = next(reader)
        headeri = {header: idx for idx, header in enumerate(headers)}
        # Process each row
        for row in reader:
            gdr = row[headeri["Gender"]]
            sal = row[headeri["Salary"]]
            # print(sal)
            e1evi = {e1_set[i]: row[headeri[e1_set[i]]] for i in range(len(e1_set))}
            e1k = tuple(sorted(e1evi.items()))
            e2k = tuple(sorted(list(e1evi.items()) + [("Gender", gdr)]))
            if e1k in res:
                probe1 = res[e1k]
            else:
                probe1 = explore_helper(bayes_net, e1evi, "Salary", ">=50K")
                # print(res)
                res[e1k] = probe1 #idk switcj orde doesnt work
            if e2k in res:
                # print(e2k)
                probe2 = res[e2k]
            else:
                e2evi = dict(e1evi)
                e2evi["Gender"] = gdr #Idk keep it as a gender variable doesnt work without
                probe2 = explore_helper(bayes_net, e2evi, "Salary", ">=50K")
                res[e2k] = probe2

            # Update counters in a single pass
            if gdr == "Female":
                ctrs['f_tot'] += 1
                # print(ctrs)
                if probe1 > probe2:
                    ctrs['f_e1_gt_e2'] += 1 #checker for e1 vs e2
                if probe1 > 0.5:
                    # print(probe1)
                    ctrs['f_e1_gt_half_tot'] += 1
                    if sal == ">=50K":
                        ctrs['f_e1_gt_half_cr'] += 1
            else:  # MALE
                ctrs['m_tot'] += 1
                if probe1 > probe2:
                    # print(probe1)
                    ctrs['m_e1_gt_e2'] += 1
                if probe1 > 0.5:
                    ctrs['m_e1_gt_half_tot'] += 1
                    if sal == ">=50K": #checker for sallary
                        ctrs['m_e1_gt_half_cr'] += 1

    # Calculate result based on question
    if question == 1:
        return (ctrs['f_e1_gt_e2'] / ctrs['f_tot']) * 100 # Prob fem >
    elif question == 2:
        return (ctrs['m_e1_gt_e2'] / ctrs['m_tot']) * 100 # Prob male >
    elif question == 3:
        return (ctrs['f_e1_gt_half_cr'] / ctrs['f_e1_gt_half_tot'] * 100
                if ctrs['f_e1_gt_half_tot'] > 0 else 0)
    elif question == 4:
        return (ctrs['m_e1_gt_half_cr'] / ctrs['m_e1_gt_half_tot'] * 100
                if ctrs['m_e1_gt_half_tot'] > 0 else 0)
    elif question == 5:
        return (ctrs['f_e1_gt_half_tot'] / ctrs['f_tot']) * 100
    elif question == 6:
        return (ctrs['m_e1_gt_half_tot'] / ctrs['m_tot']) * 100
    else:
        raise ValueError("Invalid input question number.")

def explore_helper(bnet, evars, tvar, targ):
    '''
    Helper functn for explore()
    '''
    rfactors = []
    for factor in bnet.factors():
        rfactor = factor
        for var, value in evars.items():
            vari = bnet.get_variable(var)
            if vari in rfactor.get_scope():
                rfactor = restrict(rfactor, vari, value)
        rfactors.append(rfactor)
    mfactor = multiply(rfactors)
    for var in mfactor.get_scope():
        if var.name != tvar:
            mfactor = sum_out(mfactor, var)
    return normalize(mfactor).get_value([targ])

if __name__ == '__main__':
    nb = naive_bayes_model('data/adult-train.csv')
    for i in range(1,7):
        print("explore(nb,{}) = {}".format(i, explore(nb, i)))
