#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
"""
This module provides the ScenarioManager implementation.
It must not be modified and is for reference only!
"""

from __future__ import print_function
import sys
import time

import py_trees

#from core.simulators.srunner.autoagents.agent_wrapper import AgentWrapper
from core.simulators.carla_data_provider import CarlaDataProvider
from core.simulators.srunner.scenariomanager.result_writer import ResultOutputProvider
from core.simulators.srunner.scenariomanager.timer import GameTime
from core.simulators.srunner.scenariomanager.watchdog import Watchdog


class ScenarioManager(object):
    """
    Basic scenario manager class. This class holds all functionality
    required to start, and analyze a scenario.

    The user must not modify this class.

    To use the ScenarioManager:
    1. Create an object via manager = ScenarioManager()
    2. Load a scenario via manager.load_scenario()
    3. Trigger the execution of the scenario manager.run_scenario()
       This function is designed to explicitly control start and end of
       the scenario execution
    4. Trigger a result evaluation with manager.analyze_scenario()
    5. If needed, cleanup with manager.stop_scenario()
    """

    def __init__(self, debug_mode=False, sync_mode=False, timeout=2.0):
        """
        Setups up the parameters, which will be filled at load_scenario()

        """
        self.scenario = None
        self.scenario_tree = None
        self.scenario_class = None
        self.ego_vehicles = None
        self.other_actors = None

        self._debug_mode = debug_mode
        self._sync_mode = sync_mode
        self._running = False
        self._timestamp_last_run = 0.0
        timeout = 10000
        self._timeout = timeout
        self._watchdog = Watchdog(float(self._timeout))

        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = None
        self.end_system_time = None

    def _reset(self):
        """
        Reset all parameters
        """
        self._running = False
        self._timestamp_last_run = 0.0
        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = None
        self.end_system_time = None
        GameTime.restart()

    def clean_up(self):
        """
        This function triggers a proper termination of a scenario
        """

        if self.scenario is not None:
            self.scenario.terminate()

        CarlaDataProvider.clean_up()

    def load_scenario(self, scenario):
        """
        Load a new scenario
        """
        self._reset()
        self._sync_mode = CarlaDataProvider.is_sync_mode()
        self.scenario_class = scenario
        self.scenario = scenario.scenario
        self.scenario_tree = self.scenario.scenario_tree
        self.ego_vehicles = scenario.ego_vehicles
        self.other_actors = scenario.other_actors

        # To print the scenario tree uncomment the next line
        # py_trees.display.render_dot_tree(self.scenario_tree)

    def start_scenario(self):
        print("[SCENARIO MANAGER]: Running scenario {}".format(self.scenario_tree.name))
        self.start_system_time = time.time()
        self.start_game_time = GameTime.get_time()

        if not self._debug_mode:
            self._watchdog.start()
        self._running = True

    def end_scenario(self):
        if not self._debug_mode:
            self._watchdog.stop()

        self.clean_up()

        self.end_system_time = time.time()
        end_game_time = GameTime.get_time()

        self.scenario_duration_system = self.end_system_time - \
            self.start_system_time
        self.scenario_duration_game = end_game_time - self.start_game_time

        if self.scenario_tree.status == py_trees.common.Status.FAILURE:
            print("ScenarioManager: Terminated due to failure")

    def tick_scenario(self, timestamp):
        """
        Run next tick of scenario and the agent.
        If running synchornously, it also handles the ticking of the world.
        """

        if self._timestamp_last_run < timestamp.elapsed_seconds and self._running:
            self._timestamp_last_run = timestamp.elapsed_seconds

            if not self._debug_mode:
                self._watchdog.update()

            if self._debug_mode:
                print("\n--------- Tick ---------\n")

            # Update game time and actor information
            GameTime.on_carla_tick(timestamp)
            CarlaDataProvider.on_carla_tick()

            # Tick scenario
            self.scenario_tree.tick_once()

            if self._debug_mode:
                print("\n")
                py_trees.display.print_ascii_tree(self.scenario_tree, show_status=True)
                sys.stdout.flush()

            if self.scenario_tree.status != py_trees.common.Status.RUNNING:
                self._running = False

        if self._sync_mode and self._running and self._watchdog.get_status():
            CarlaDataProvider.get_world().tick()

    def get_running_status(self):
        """
        returns:
           bool:  False if watchdog exception occured, True otherwise
        """
        return self._watchdog.get_status()

    def is_running(self):
        return self._running

    def stop_scenario(self):
        """
        This function is used by the overall signal handler to terminate the scenario execution
        """
        self._running = False

    def analyze_scenario(self, stdout, filename, junit):
        """
        This function is intended to be called from outside and provide
        the final statistics about the scenario (human-readable, in form of a junit
        report, etc.)
        """

        failure = False
        timeout = False
        result = "SUCCESS"

        if self.scenario.test_criteria is None:
            print("Nothing to analyze, this scenario has no criteria")
            return True

        for criterion in self.scenario.get_criteria():
            if (not criterion.optional and criterion.test_status != "SUCCESS"
                    and criterion.test_status != "ACCEPTABLE"):
                failure = True
                result = "FAILURE"
            elif criterion.test_status == "ACCEPTABLE":
                result = "ACCEPTABLE"

        if self.scenario.timeout_node.timeout and not failure:
            timeout = True
            result = "TIMEOUT"

        output = ResultOutputProvider(self, result, stdout, filename, junit)
        output.write()

        return failure or timeout

    def analyze_tick(self):
        if self.scenario.test_criteria is None:
            print("Nothing to analyze, this scenario has no criteria")

        criteria_list = []

        for criterion in self.scenario.get_criteria():
            name_string = criterion.name
            actor_id = criterion.actor.id
            result = criterion.test_status
            actual_value = criterion.actual_value
            expected_value = criterion.expected_value_success
            criteria_list.append([name_string, actor_id, result, actual_value, expected_value])
        return criteria_list

    def get_scenario_status(self):
        if self.scenario_tree.status == py_trees.common.Status.RUNNING:
            res = 'RUNNING'
        elif self.scenario_tree.status == py_trees.common.Status.SUCCESS:
            res = 'SUCCESS'
        elif self.scenario_tree.status == py_trees.common.Status.FAILURE:
            res = 'FAILURE'
        else:
            res = 'INVALID'
        return res
