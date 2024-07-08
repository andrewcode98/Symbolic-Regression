# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 08:16:45 2024

@author: andre
"""

import math
import random
import scipy.stats as stats
import numpy as np
import sqlite3
import re
import pandas as pd
from tqdm import tqdm
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed





def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False
    
def is_variable(s):
    # Define the pattern: a letter (from a to z or A to Z) followed by one or more digits
    pattern = r'^[a-zA-Z]([0-9]+)?$'
    
    # Use re.match to check if the string matches the pattern
    return bool(re.match(pattern, s))
    
def is_operator(token):
    return token in {'+', '-', '*', '/', '^'}

def is_function(token):
    return token in {'sin', 'cos', 'atan', 'exp', 'log', 'norm', 'sqrt'}


class Node:
    def __init__(self, value):
        # Evaluate node class with left and right children
        self.value = value
        self.left = None
        self.right = None

class Binary_Tree:
    # Initialise binary tree with postfix notation, root and operators(both binary and unary)
    def __init__(self, postfix):
        self.original_postfix = postfix  # Store the original postfix expression
        self.postfix = postfix[:]  # Make a copy of the postfix expression
        self.root = None
        self.operators = {'+': lambda x, y: x + y,
                          '-': lambda x, y: x - y,
                          '*': lambda x, y: x * y,
                          '/': lambda x, y: x / y,
                          '^': lambda x, y: x ** y}
        self.unary_operators = {'sin': lambda x: math.sin(x),
                                'cos': lambda x: math.cos(x),
                                'atan': lambda x: math.atan(x),
                                'exp': lambda x: math.exp(x),
                                'log': lambda x: math.log(x),
                                'norm': lambda x: stats.norm.cdf(x),
                                'sqrt': lambda x: math.sqrt(x)}
        self.variables = []
        stack = []
        for element in self.postfix:
            if is_float(element) and element not in stack: 
                self.variables.append(element)
                stack.append(element)
            
    def build_tree(self):
        stack = []
        
        
        
        
        
        for token in self.postfix:
            # If we have a feature or constant add it to stack
            if is_float(token) or is_variable(token):  # Operand (number)
                node = Node(float(token))  
                stack.append(node)
            # If its a unary operator we remove the last node from the stack and set it
            # as the left child of our current operator, add the new operator to the stack
            elif token in self.unary_operators: 
                node = Node(token)
                operand = stack.pop()
                node.left = operand
                stack.append(node)
            # For binary operator we remove the last and 2nd to last elements from the stack
            # and add them as left and right children to the binary operator, add the 
            # binary operator to the stack.
            elif token in self.operators:  
                node = Node(token)
                right = stack.pop()
                left = stack.pop()
                node.left = left
                node.right = right
                stack.append(node)
        
        # The root of the binary tree is the last node in the stack
        self.root = stack.pop() if stack else None
        
        # Evaluate the result of the tree starting from the root
        if self.root:
            return self._evaluate(self.root)
        else:
            return None
    
    def _evaluate(self, node):
        try:
            if node:
                if node.value in self.operators:
                    # Recursively evaluate the smaller trees starting from the root
                    left_val = self._evaluate(node.left)
                    right_val = self._evaluate(node.right)
                    return self.operators[node.value](left_val, right_val)
                elif node.value in self.unary_operators:
                    # Evaluate unary operator
                    operand_val = self._evaluate(node.left)
                    return self.unary_operators[node.value](operand_val)
                else:
                    # Feature or constant
                    return node.value
        except Exception:
            # Return NaN to indicate an error
            return float('nan')

    def generate_postfix(self):
        postfix = []

        def traverse(node):
            nonlocal postfix
            if node:
                traverse(node.left)
                traverse(node.right)
                postfix.append(str(node.value))
    
        traverse(self.root)
        self.postfix = postfix
    
    def print_tree(self):
        self._print_node(self.root, 0)
    
    def _print_node(self, node, depth):
        if node:
            
            if node.right:
                self._print_node(node.right, depth + 1)
            
            # Print node value at the current depth
            print(" " * 4 * depth + "--> " + str(node.value))
            
            # Recursively print left subtree
            if node.left:
                self._print_node(node.left, depth + 1)
 


    

        
   


def postfix_to_infix(tokens):
    
    
    
    
    stack = []
    
    
    
    for token in tokens:
        if is_float(token) or is_variable(token):
            # If the token is an operand (number), push it to the stack
            stack.append(token)
        elif is_function(token):
            # If the token is a function, pop one operand from the stack, apply the function and push the result back
            operand = stack.pop()
            stack.append(f"{token}({operand})")
        elif is_operator(token):
            # If the token is an operator, pop two operands from the stack, apply the operator and push the result back
            right_operand = stack.pop()
            left_operand = stack.pop()
            result = f"({left_operand} {token} {right_operand})"
            stack.append(result)
    
    # The result should be the only element left in the stack
    return stack[0]



 
# Define possible binary operators
binary_operators = ['-','*']    
operands = ['d1', 'd2', 'S' , 'K' , 'r', 'T']
# unary_operators = ['sin', 'cos', 'atan', 'exp', 'log', 'norm', 'sqrt']
unary_operators = ['norm','exp']

def generate_postfix_permutations(operands, unary_operators, binary_operators, desired_length, sample_size):
    elements = operands + unary_operators + binary_operators
    
    # Define probabilities for each type of element
    operand_prob = 6/14
    unary_operator_prob = 3/14
    binary_operator_prob = 5/14
    
    # Create a list of weights for each element in the `elements` list
    weights = (
        [operand_prob / len(operands)] * len(operands) +
        [unary_operator_prob / len(unary_operators)] * len(unary_operators) +
        [binary_operator_prob / len(binary_operators)] * len(binary_operators)
    )

    def is_valid_postfix(perm):
        stack_depth = 0
        for elem in perm:
            if elem in operands:
                stack_depth += 1
            elif elem in unary_operators:
                if stack_depth < 1:
                    return False
            elif elem in binary_operators:
                if stack_depth < 2:
                    return False
                stack_depth -= 1
        return stack_depth == 1

    def are_operands_repeated(perm):
        operand_count = [elem for elem in perm if elem in operands]
        return len(set(operand_count)) != len(operand_count)

    valid_postfix_expressions = set()
    desired_length = random.randint(1, desired_length)
    attempts = 0  # to avoid infinite loop in case of too restrictive conditions
    while len(valid_postfix_expressions) < sample_size and attempts < sample_size:
        perm = tuple(np.random.choice(elements, size=desired_length, p=weights))
        if not are_operands_repeated(perm) and is_valid_postfix(perm):
            valid_postfix_expressions.add(perm)
        attempts += 1

       
    return [list(expr) for expr in valid_postfix_expressions]  # Convert back to list

# Example usage
operands = ['d1', 'd2', 'S' , 'K' , 'r', 'T']
unary_operators = ['norm','exp']
binary_operators = ['-', '*']
desired_length = 14
sample_size = 1000

np.random.seed(0)
num_options = 100
S0 = np.random.uniform(80, 120, num_options)  # Initial stock price
K = np.random.uniform(80, 120, num_options)  # Strike prices
T = np.random.uniform(0.1, 1, num_options)  # Time to maturity
r = np.random.uniform(0.1, 0.8, num_options) # Risk-free rate
sigma = np.random.uniform(0.1, 0.4, num_options)  # Volatility

# Black-Scholes formula for European call option price
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    return call_price

# Generate synthetic data
call_prices = black_scholes_call(S0, K, T, r, sigma)
data = pd.DataFrame({
    'S': S0,
    'K': K,
    'T': T,
    'r': -r,
    'd1': (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T)),
    'd2': (np.log(S0 / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T)),
    'CallPrice': call_prices
})



# Convert DataFrame 
X = data[['S', 'K', 'T', 'r', 'd1', 'd2']]
y = data['CallPrice']


    
def replace_operands_in_expressions(expressions, X):
    replaced_expressions = []
    
    def replace_operands_for_row(row):
        replaced_row_expressions = []
        for expr in expressions:
            replaced_expr = [row[operand] if operand in row else operand for operand in expr]
            replaced_row_expressions.append(replaced_expr)
        return replaced_row_expressions
    
    # Use ThreadPoolExecutor for concurrent processing
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(replace_operands_for_row, row) for _, row in X.iterrows()]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Replacing operands in expressions"):
            replaced_expressions.append(future.result())
    
    return replaced_expressions


def mean_absolute_error(predicted, actual):
    return np.mean(np.abs((actual - predicted) / actual))

def get_different_choice(current, choices):
    new_choice = random.choice([choice for choice in choices if choice != current])
    return new_choice

def contains_invalid_sequence(expression, invalid_sequences):
    """ Check if the expression contains any of the invalid sequences """
    expr_str = ' '.join(expression)  # Convert list to string for easier searching
    for seq in invalid_sequences:
        seq_str = ' '.join(seq)
        if seq_str in expr_str:
            return True
    return False

def generate_mutated_expressions(initial, operands, unary_operators, binary_operators, constraints):
    mutated_expressions = []

    # Define the invalid sequences
    

    for expr in initial:
        new_expr = expr.copy()  # Create a copy of the current expression

        index = random.randint(0, len(new_expr) - 1)  # Select a random element to replace
        operand_positions = [index for index, element in enumerate(new_expr) if element in operands]

        # Switch the variables positions
        if new_expr[index] in operands:
            if len(operand_positions) > 1:
                pos2 = random.choice([pos for pos in operand_positions if pos != index])
                new_expr[index], new_expr[pos2] = new_expr[pos2], new_expr[index]
        elif new_expr[index] in unary_operators:
            new_expr[index] = get_different_choice(new_expr[index], unary_operators)
        elif new_expr[index] in binary_operators:
            new_expr[index] = get_different_choice(new_expr[index], binary_operators)

        # Append the mutated expression to the new list if it does not contain an invalid sequence
        if not contains_invalid_sequence(new_expr, constraints):
            mutated_expressions.append(new_expr)

    return mutated_expressions + initial


def create_database(db_file):
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS expressions
                 (expression TEXT PRIMARY KEY, mape REAL)''')
    conn.commit()
    conn.close()

def load_expressions_from_db(db_file):
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute('SELECT expression FROM expressions')
    rows = c.fetchall()
    conn.close()
    return set(row[0] for row in rows)

def save_expressions_to_db(db_file, expressions):
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    # Convert expressions to tuples of (expression_str, mape)
    expressions_to_save = [(str(expr[0]), expr[1]) for expr in expressions]
    c.executemany('INSERT OR IGNORE INTO expressions VALUES (?, ?)', expressions_to_save)
    conn.commit()
    conn.close()
    
def evaluate_expression(expression):
    return Binary_Tree(expression).build_tree()
        
def find_hall_of_fame(operands, unary_operators, binary_operators, desired_length,
                      sample_size, X, y,
                      num_mutations, mape_threshold, constraints,
                      db_file='expressions.db'):
    
    create_database(db_file)
    evaluated_expressions_set = load_expressions_from_db(db_file)

    # Generate valid postfix expressions
    valid_postfix_expressions = generate_postfix_permutations(operands, unary_operators, binary_operators, desired_length, sample_size)
    
    for _ in range(num_mutations):
        valid_postfix_expressions = generate_mutated_expressions(valid_postfix_expressions, operands,
                                                                 unary_operators, binary_operators, constraints)
    
    # Filter out already evaluated expressions
    valid_postfix_expressions = [expr for expr in valid_postfix_expressions if str(expr) not in evaluated_expressions_set]
    
    print("Expressions to be evaluated: ", len(valid_postfix_expressions))
    
    # Replace operands in expressions with actual float values from X
    replaced_expressions = replace_operands_in_expressions(valid_postfix_expressions, X)
    
    # Evaluate each expression and compute MAPE
    hall_of_fame = []
    list_of_expressions = []
    
    mape = np.empty(len(valid_postfix_expressions), dtype=float)
    results = np.empty((len(X), len(valid_postfix_expressions)), dtype=float)
    
    # Flatten the tasks to parallelize
    tasks = [(i, j) for i in range(len(valid_postfix_expressions)) for j in range(len(X))]
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_task = {executor.submit(evaluate_expression, replaced_expressions[j][i]): (j, i) for i, j in tasks}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(tasks), desc="Evaluating expressions"):
            j, i = future_to_task[future]
            try:
                results[j][i] = future.result()
            except Exception as exc:
                print(f'Task {(j, i)} generated an exception: {exc}')
    
    # Calculate MAPE and update hall of fame
    for i in range(len(valid_postfix_expressions)):
        mape[i] = np.mean(np.abs((y - results[:, i]) / y))
        
        # Add to hall of fame if MAPE < threshold
        if mape[i] < mape_threshold:
            hall_of_fame.append((valid_postfix_expressions[i], mape[i]))
        list_of_expressions.append((valid_postfix_expressions[i], mape[i]))
    
    # Save evaluated expressions to database
    save_expressions_to_db(db_file, list_of_expressions)
    
    return hall_of_fame


mape_threshold = 0.05
constraints = [
    ['norm', 'norm'],
    ['norm', 'exp'],
    ['exp', 'norm'],
    ['exp', 'exp']
]
hall_of_fame = find_hall_of_fame(operands, unary_operators, binary_operators, desired_length,
                                 sample_size, X, y, 10, mape_threshold,
                                 constraints)

# Print hall of fame equations along with their MAPE values
print(" ")
print("Hall of fame equations:")
for expr, mape_value in hall_of_fame:
    infix_expr = postfix_to_infix(expr)
    
    print(f"Equation: {infix_expr}, MAPE: {mape_value:.5f}")
    
#print(postfix_to_infix(['S','d1','norm','*','K','r','T','*','exp','d2','norm','*','*','-']))
   
    
