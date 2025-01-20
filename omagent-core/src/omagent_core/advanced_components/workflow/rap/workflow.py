from omagent_core.engine.workflow.conductor_workflow import ConductorWorkflow
from omagent_core.engine.workflow.task.simple_task import simple_task
from omagent_core.engine.workflow.task.do_while_task import DoWhileTask

class RAPWorkflow(ConductorWorkflow):
    def __init__(self):
        super().__init__(name="rap_workflow")

    def set_input(self, query: str):
        self.query = query
        self._configure_tasks()
        self._configure_workflow()

    def _configure_tasks(self):
        # Input interface task
        self.input_task = simple_task(
            task_def_name='InputInterface',
            task_reference_name='input_task',
            inputs={"query": self.query}
        )

        # Selection task
        self.selection_task = simple_task(
            task_def_name='Selection',
            task_reference_name='selection'
        )

        # Expansion task
        self.expansion_task = simple_task(
            task_def_name='Expansion',
            task_reference_name='expansion'
        )

        # Simulation pre-process task
        self.simulation_preprocess_task = simple_task(
            task_def_name='SimulationPreProcess',
            task_reference_name='simulation_preprocess'
        )

        # Simulation expansion task
        self.expansion2_task = simple_task(
            task_def_name='Expansion',
            task_reference_name='expansion2'
        )

        # Simulation post-process task
        self.simulation_postprocess_task = simple_task(
            task_def_name='SimulationPostProcess',
            task_reference_name='simulation_postprocess'
        )

        # Back propagation task
        self.back_propagation_task = simple_task(
            task_def_name='BackPropagation',
            task_reference_name='back_propagation'
        )

        # MCTS completion check task
        self.mcts_completion_check_task = simple_task(
            task_def_name='MCTSCompletionCheck',
            task_reference_name='mcts_completion_check'
        )

        # Output interface task
        self.output_interface_task = simple_task(
            task_def_name='OutputInterface',
            task_reference_name='output_interface'
        )

        # Configure simulation loop
        self.simulation_loop = DoWhileTask(
            task_ref_name='simulation_loop',
            tasks=[self.expansion2_task, self.simulation_postprocess_task],
            termination_condition='if ($.simulation_postprocess["finish"] == true){false;} else {true;}'
        )

        # Configure MCTS loop
        self.mcts_loop = DoWhileTask(
            task_ref_name='mcts_loop',
            tasks=[
                self.selection_task,
                self.expansion_task,
                self.simulation_preprocess_task,
                self.simulation_loop,
                self.back_propagation_task,
                self.mcts_completion_check_task
            ],
            termination_condition='if ($.mcts_completion_check["finish"] == true){false;} else {true;}'
        )

    def _configure_workflow(self):
        # Configure workflow execution flow
        self >> self.input_task >> self.mcts_loop >> self.output_interface_task
        
        # Set outputs
        self.rap_structure = self.output_interface_task.output("rap_structure")
        self.final_answer = self.output_interface_task.output("final_answer") 