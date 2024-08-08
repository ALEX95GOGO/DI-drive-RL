
from __future__ import print_function

import math
import py_trees
import carla

from core.simulators.carla_data_provider import CarlaDataProvider

from core.simulators.srunner.scenarios.basic_scenario import BasicScenario
from core.simulators.srunner.tools.scenario_helper import get_waypoint_in_distance, get_location_in_distance_from_wp
from core.simulators.srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorDestroy,
                                                                      KeepVelocity,
                                                                      Idle,
                                                                      WaypointFollower,
                                                                      ActorTransformSetter,
                                                                      SyncArrival,
                                                                      ActorSink)
from core.simulators.srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (InTriggerDistanceToLocation,
                                                                               InTimeToArrivalToLocation,
                                                                               DriveDistance, InTriggerDistanceToLocation,
                                                                               InTriggerDistanceToVehicle)
from core.utils.simulator_utils.carla_agents.navigation.global_route_planner import GlobalRoutePlanner
from core.utils.simulator_utils.carla_agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from core.utils.simulator_utils.carla_agents.navigation.local_planner import RoadOption

import numpy as np

class OccludedLeftCrossingJunction(BasicScenario):
    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=120):
        self._wmap = CarlaDataProvider.get_map()
        self._reference_waypoint = self._wmap.get_waypoint(config.trigger_points[0].location)
        self._ego_route = CarlaDataProvider.get_ego_vehicle_route()
        self.timeout = timeout

        self._occluder_wait_time = 1.5
        self._occluder_distance_threshold = 12
        self._occluder_speed = 7
        self._occluder_driving_distance = 20
        self._occluder_prev_distance = 15
        self._destory_actors_threshold = 8

        self._blocker_speed = 2
        self._blocker_distance_threshold = 6

        super(OccludedLeftCrossingJunction, self).__init__("LeftCuttingInStationaryObject",
                                                    ego_vehicles,
                                                    config,
                                                    world,
                                                    debug_mode,
                                                    criteria_enable=criteria_enable)
        
        
        #if ego_vehicles[0].is_at_traffic_light():
        #    traffic_light = ego_vehicles[0].get_traffic_light()
        #    if traffic_light.get_state() == carla.TrafficLightState.Red:
        #       # world.hud.notification("Traffic light changed! Good to go!")
        #        traffic_light.set_state(carla.TrafficLightState.Green)
        
        #self._traffic_light = CarlaDataProvider.get_next_traffic_light_by_location(config.trigger_points[0].location)
        #light_manager = CarlaDataProvider._world.get_lightmanager()
        #self._traffic_light = light_manager.get_all_traffic_lights()
        
        list_actor = CarlaDataProvider._world.get_actors()
        for actor_ in list_actor:
            if isinstance(actor_, carla.TrafficLight):
                # for any light, first set the light state, then set time. for yellow it is 
                # carla.TrafficLightState.Yellow and Red it is carla.TrafficLightState.Red
                actor_.set_state(carla.TrafficLightState.Green) 
                actor_.set_green_time(1000.0)

        #if self._traffic_light is not None:
        #    self._traffic_light.set_state(carla.TrafficLightState.Green)
        #    self._traffic_light.set_green_time(self.timeout)


    def _interpolate_trajectory(self, start_waypoint, end_waypoint, hop_resolution=1.0):
        dao = GlobalRoutePlannerDAO(self._wmap, hop_resolution)
        grp = GlobalRoutePlanner(dao)
        grp.setup()

        plans = []
        interpolated_trace = grp.trace_route(start_waypoint.transform.location, end_waypoint.transform.location)
        for wp_tuple in interpolated_trace:
            plans.append((wp_tuple[0], wp_tuple[1]))
        return plans

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        waypoint = self._reference_waypoint

        self._blocker_waypoint = self._wmap.get_waypoint(config.other_actors[0].transform.location)
        blocker_location = carla.Location(config.other_actors[0].transform.location.x,
                                          config.other_actors[0].transform.location.y,
                                          config.other_actors[0].transform.location.z - 500)
        blocker_transform = carla.Transform(blocker_location, self._blocker_waypoint.transform.rotation)
        blocker = CarlaDataProvider.request_new_actor('vehicle.carlamotors.carlacola', blocker_transform)
        blocker.set_simulate_physics(enabled=False)

        self._occluder_waypoint = self._blocker_waypoint.get_right_lane().previous(self._occluder_prev_distance)[0]
        occluder_location = carla.Location(self._occluder_waypoint.transform.location.x,
                                          self._occluder_waypoint.transform.location.y,
                                          self._occluder_waypoint.transform.location.z - 500)
        occluder_transform = carla.Transform(occluder_location, self._occluder_waypoint.transform.rotation)
        occluder = CarlaDataProvider.request_new_actor('vehicle.audi.tt', occluder_transform)

        occluder.set_simulate_physics(enabled=False)
        self.other_actors.append(occluder)
        self.other_actors.append(blocker)

        self._blocker_target_location = config.other_actors[1].transform.location
        blocker_target_waypoint = self._wmap.get_waypoint(self._blocker_target_location)
        self._blocker_plans = self._interpolate_trajectory(self._blocker_waypoint, blocker_target_waypoint)

    def _create_behavior(self):
        """
        After invoking this scenario, vehicle will for the user
        controlled vehicle to enter trigger distance region,
        the vehicle starts crossing the road from the
        stationary vehicles before the junction once the condition meets,
        then after 120 seconds, a timeout stops the scenario
        """
        sequence = py_trees.composites.Sequence()

        sink_parallel = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        sink_parallel.add_child(Idle(0.5))
        sink_parallel.add_child(ActorSink(self._occluder_waypoint.transform.location, self._destory_actors_threshold))
        sink_parallel.add_child(ActorSink(self._blocker_waypoint.transform.location, self._destory_actors_threshold))
        sequence.add_child(sink_parallel)

        sequence.add_child(ActorTransformSetter(self.other_actors[0], self._occluder_waypoint.transform))
        sequence.add_child(ActorTransformSetter(self.other_actors[1], self._blocker_waypoint.transform))

        collision_location = self._get_collision_location(self._occluder_waypoint)

        runningcondition = py_trees.composites.Parallel("running all the vehicles", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)

        blocker_runningcondition = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        blocker_runningcondition.add_child(WaypointFollower(self.other_actors[1], target_speed=self._blocker_speed, plan = self._blocker_plans, avoid_collision=True))
        blocker_runningcondition.add_child(InTriggerDistanceToLocation(self.other_actors[1], self._blocker_target_location, self._blocker_distance_threshold))

        runningcondition.add_child(blocker_runningcondition)

        occluder_runningcondition = py_trees.composites.Sequence()
        occluder_runningcondition.add_child(Idle(self._occluder_wait_time))

        occluder_start = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        occluder_start.add_child(SyncArrival(self.other_actors[0], self.ego_vehicles[0], collision_location))
        occluder_start.add_child(InTriggerDistanceToLocation(self.other_actors[0], collision_location, self._occluder_distance_threshold))

        occluder_runningcondition.add_child(occluder_start)

        occluder_runningcondition.add_child(KeepVelocity(self.other_actors[0], self._occluder_speed, distance=self._occluder_driving_distance))

        runningcondition.add_child(occluder_runningcondition)

        sequence.add_child(runningcondition)
        sequence.add_child(ActorDestroy(ActorDestroy(self.other_actors[0])))
        sequence.add_child(ActorDestroy(ActorDestroy(self.other_actors[1])))

        return sequence

    def _get_collision_location(self, waypoint):
        self._ego_route = CarlaDataProvider.get_ego_vehicle_route()
        current_location = self._reference_waypoint.transform.location

        min_id = None
        min_distance = 100000

        for i, node in enumerate(self._ego_route):
            #import pdb; pdb.set_trace()
            distance = node[0].location.distance(current_location)
            if distance < min_distance:
                min_distance = distance
                min_id = i
        
        #import pdb; pdb.set_trace()
        # skip the waypoints not in the intersection
        while not self._wmap.get_waypoint(self._ego_route[min_id][0].location).is_junction:
            min_id = min_id + 1
        
        # get the farest waypoint in the intersection
        max_id = min_id + 1
        while self._wmap.get_waypoint(self._ego_route[max_id][0].location).is_junction:
            max_id = max_id + 1

        _min_id = None
        _min_distance = 100000
        reference_location = waypoint.transform.location
        for i in range(min_id, max_id + 1):
            distance = reference_location.distance(self._ego_route[i][0].location)
            if distance < _min_distance:
                _min_distance = distance
                _min_id = i
        return self._ego_route[_min_id][0]

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        collision_criterion = CollisionTest(self.ego_vehicles[0])
        criteria.append(collision_criterion)

        return criteria

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()

