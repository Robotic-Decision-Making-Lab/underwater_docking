cmake_minimum_required(VERSION 3.8)
project(example_pkg)

set(THIS_PACKAGE_INCLUDE_DEPENDS
  catkin
)

foreach(Dependency IN ITEMS ${THIS_PACKAGE_INCLUDE_DEPENDS})
  find_package(${Dependency} REQUIRED)
endforeach()
