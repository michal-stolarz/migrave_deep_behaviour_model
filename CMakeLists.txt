cmake_minimum_required(VERSION 3.0.2)
project(migrave_deep_behaviour_model)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  genmsg
  std_msgs
  geometry_msgs
  sensor_msgs
)

#add_message_files(
#   DIRECTORY msg
#   FILES
#   GameAction.msg
#)

#add_action_files(
#  DIRECTORY action
#  FILES
#
#)

#add_service_files(
#  DIRECTORY srv
#  FILES
#
#)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  sensor_msgs
)

catkin_package(
   CATKIN_DEPENDS
     message_runtime
     std_msgs
     geometry_msgs
     sensor_msgs
)
