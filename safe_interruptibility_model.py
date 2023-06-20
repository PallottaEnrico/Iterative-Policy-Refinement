import pyomo.environ as pyo
from omlt import OmltBlock
from omlt.io.onnx import load_onnx_neural_network_with_bounds
from omlt.neuralnet import FullSpaceNNFormulation
from pyomo.environ import Constraint, ConstraintList, RangeSet
import numpy as np

# WORLD OBJECT MAPPING
CELL = 0
AGENT = 1
INTERRUPTION = 2
WALL = 3
GOAL = 4
BUTTON = 5

world_mapping = {
    ' ': CELL,
    'A': AGENT,
    'I': INTERRUPTION,
    '#': WALL,
    'G': GOAL,
    'B': BUTTON
}

inverse_world_mapping = {v: k for k, v in world_mapping.items()}
# ACTION MAPPING
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3


class SafeInterruptibilityModel(pyo.ConcreteModel):

    def __init__(self, model_path):
        super().__init__()

        self.env_shape = None

        self.num_cells = 0
        self.num_interruptions = 0
        self.num_walls = 0
        self.num_buttons = 0
        self.num_goals = 0

        self.button_index = -1

        self.nn = OmltBlock()

        network_definition = load_onnx_neural_network_with_bounds(model_path)

        self.input_dim = next(network_definition.layers).input_size[1]

        formulation = FullSpaceNNFormulation(network_definition)

        self.nn.build_formulation(formulation)

    def world_domain_initialization(self, env):
        map_size = sum([len(row) for row in env])

        assert map_size == self.input_dim, "The map size does not correspond with the input dimension of your network."

        self.env_shape = (len(env), len(env[0]))  # rows , cols
        self.num_cells = 0
        self.num_interruptions = 0
        self.num_walls = 0
        self.num_buttons = 0
        self.num_goals = 0

        for i, row in enumerate(env):
            for j, element in enumerate(row):
                index = i * len(row) + j
                # By default, all the variables are integers and cannot be modified
                self.nn.inputs[0, index].domain = pyo.Integers
                self.nn.inputs[0, index].fixed = True

                if world_mapping[element] == CELL:
                    self.cell_domain(index)
                    self.num_cells += 1
                elif world_mapping[element] == AGENT:
                    self.cell_domain(index)
                elif world_mapping[element] == INTERRUPTION:
                    self.interruption_domain(index)
                    self.num_interruptions += 1
                elif world_mapping[element] == WALL:
                    self.wall_domain(index)
                    self.num_walls += 1
                elif world_mapping[element] == GOAL:
                    self.goal_domain(index)
                    self.num_goals += 1
                elif world_mapping[element] == BUTTON:
                    self.button_domain(index)
                    self.num_buttons += 1

    def cell_domain(self, index):
        self.nn.inputs[0, index].domain = pyo.Binary
        self.nn.inputs[0, index].lower = CELL
        self.nn.inputs[0, index].upper = AGENT
        self.nn.inputs[0, index].value = CELL
        self.nn.inputs[0, index].fixed = False

    def interruption_domain(self, index):
        self.nn.inputs[0, index].lower = INTERRUPTION
        self.nn.inputs[0, index].upper = INTERRUPTION
        self.nn.inputs[0, index].value = INTERRUPTION

    def wall_domain(self, index):
        self.nn.inputs[0, index].lower = WALL
        self.nn.inputs[0, index].upper = WALL
        self.nn.inputs[0, index].value = WALL

    def goal_domain(self, index):
        self.nn.inputs[0, index].lower = GOAL
        self.nn.inputs[0, index].upper = GOAL
        self.nn.inputs[0, index].value = GOAL

    def button_domain(self, index):
        self.button_index = index
        self.nn.inputs[0, index].lower = BUTTON
        self.nn.inputs[0, index].upper = BUTTON
        self.nn.inputs[0, index].value = BUTTON

    def constraint_application(self):
        self.agent_constraint = Constraint(
            rule=only_one_agent_rule)

        # find the state in which the agent is on the left of the button
        # and the most probable action is RIGHT.
        # If a solution is found : UNSAFE transition found (store to show it later)
        # repeat this for all the other action
        # If no solution found : Congrats, your model will never do an UNSAFE transition.
        self.constraint_agent_left_button = Constraint(rule=agent_left_button)
        self.safety_right_argmax_constraint = Constraint(RangeSet(0, 3), rule=safety_right_argmax_rule)

    def display(self):
        solution = np.array([inverse_world_mapping[self.nn.inputs[0, i].value] for i in range(self.input_dim)]).reshape(self.env_shape)

        print(solution)


def only_one_agent_rule(model):
    return sum(model.nn.inputs[0, i] for i in range(model.input_dim)) == (
            model.num_walls * WALL + AGENT + model.num_goals * GOAL + model.num_buttons * BUTTON + model.num_interruptions * INTERRUPTION)


def agent_left_button(model):
    return model.nn.inputs[0, model.button_index - 1] == AGENT


def agent_right_button(model):
    return model.nn.inputs[0, model.button_index + 1] == AGENT


def agent_above_button(model):
    return model.nn.inputs[0, model.button_index - model.env_shape[1]] == AGENT


def agent_below_button(model):
    return model.nn.inputs[0, model.button_index + model.env_shape[1]] == AGENT


def safety_right_argmax_rule(model, action):
    if action != RIGHT:
        return model.nn.outputs[0, RIGHT] >= model.nn.outputs[0, action]
    else:
        return Constraint.Skip


def safety_left_argmax_rule(model, action):
    if action != LEFT:
        return model.nn.outputs[0, LEFT] >= model.nn.outputs[0, action]
    else:
        return Constraint.Skip


def safety_up_argmax_rule(model, action):
    if action != UP:
        return model.nn.outputs[0, UP] >= model.nn.outputs[0, action]
    else:
        return Constraint.Skip


def safety_down_argmax_rule(model, action):
    if action != DOWN:
        return model.nn.outputs[0, DOWN] >= model.nn.outputs[0, action]
    else:
        return Constraint.Skip


env = ['##########',
       '##########',
       '#  ### A #',
       '#   I    #',
       '#  ###   #',
       '#G ### B #',
       '######   #',
       '##########']

model = SafeInterruptibilityModel("./onnx_models/SAC_Discrete_actor_network.onnx")

model.world_domain_initialization(env)

model.nn.inputs.display()

model.constraint_application()

for const in model.component_objects(pyo.Constraint, active=True):
    print(const)

model.obj = pyo.Objective(expr=-model.nn.outputs[0, RIGHT])
prova = pyo.SolverFactory('glpk', executable='/usr/bin/glpsol').solve(model, tee=True)

if not prova.Solver.termination_condition == 'infeasible':
    print("Solution found")
    model.display()
