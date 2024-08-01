cmake_minimum_required(VERSION 3.28)

project("LearnVulkanDemos_00_HelloWindow" VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

include(NoInSourceBuilds)
CheckNoInSourceBuilds()

find_package(Vulkan REQUIRED)

add_subdirectory(external/glfw-3.4)
add_subdirectory(external/glm-1.0.1)
add_subdirectory(external/fmt-10.2.0)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")

add_executable(LearnVulkanDemos_00_HelloWindow)
target_sources(LearnVulkanDemos_00_HelloWindow PRIVATE
    src/main.cpp
)
target_link_libraries(LearnVulkanDemos_00_HelloWindow glfw)
target_link_libraries(LearnVulkanDemos_00_HelloWindow glm)
target_link_libraries(LearnVulkanDemos_00_HelloWindow fmt)
target_link_libraries(LearnVulkanDemos_00_HelloWindow Vulkan::Vulkan)

add_custom_target(run
    COMMAND ${CMAKE_COMMAND} -E env $<TARGET_FILE:LearnVulkanDemos_00_HelloWindow>
    DEPENDS "LearnVulkanDemos_00_HelloWindow"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    VERBATIM
)
