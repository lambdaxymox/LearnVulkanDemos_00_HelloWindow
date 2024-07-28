#include <vulkan/vulkan.h>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <vector>
#include <optional>
#include <set>
#include <cstdint>
#include <limits>
#include <algorithm>

#include <fmt/core.h>
#include <fmt/ostream.h>

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

const char* VK_LAYER_KHRONOS_validation = "VK_LAYER_KHRONOS_validation";
const char* VK_KHR_portability_subset = "VK_KHR_portability_subset";

const std::vector<const char*> VALIDATION_LAYERS = { 
    VK_LAYER_KHRONOS_validation
};

const std::vector<const char*> DEVICE_EXTENSIONS = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

#ifdef NDEBUG
const bool ENABLE_VALIDATION_LAYERS = false;
#else
const bool ENABLE_VALIDATION_LAYERS = true;
#endif

const int MAX_FRAMES_IN_FLIGHT = 2;

namespace vk_platform {
    enum class Platform {
        Apple,
        Linux,
        Windows,
        Unknown,
    };

    constexpr Platform detectOperatingSystem() {
        #if defined(__APPLE__) || defined(__MACH__)
        return Platform::Apple;
        #elif defined(__LINUX__)
        return Platform::Linux;
        #elif defined(_WIN32)
        return Platform::Windows;
        #else
        return Platform::Unknown;
        #endif
    }
}

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};


class App {
    public:
        void run() {
            this->createGLFWLibrary();
            this->createWindow();
            this->initVulkan();
            this->mainLoop();
            this->cleanup();
        }
    private:
        GLFWwindow* m_window;
        VkInstance m_instance;
        VkDebugUtilsMessengerEXT m_debugMessenger;
        VkSurfaceKHR m_surface;

        VkPhysicalDevice m_physicalDevice;
        VkDevice m_device;
        VkQueue m_graphicsQueue;
        VkQueue m_presentQueue;

        VkSwapchainKHR m_swapChain;
        std::vector<VkImage> m_swapChainImages;
        VkFormat m_swapChainImageFormat;
        VkExtent2D m_swapChainExtent;
        std::vector<VkImageView> m_swapChainImageViews;
        
        
        void createGLFWLibrary() {
            const auto result = glfwInit();
            if (!result) {
                glfwTerminate();

                auto errorMessage = std::string { "Failed to initialize GLFW" };

                throw std::runtime_error { errorMessage };
            }
        }

        void enumerateExtensions() {
            uint32_t extensionCount = 0;
            vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

            auto extensions = std::vector<VkExtensionProperties> { extensionCount };
            vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());

            for (const auto& extension: extensions) {
                fmt::println("NAME: {} ; VERSION: {}", extension.extensionName, extension.specVersion);
            }
        }

        std::vector<const char*> getRequiredExtensions() {
            uint32_t glfwExtensionCount = 0;
            const char** glfwExtensions;
            glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

            auto requiredExtensions = std::vector<const char*> { glfwExtensions, glfwExtensions + glfwExtensionCount };
            if (ENABLE_VALIDATION_LAYERS) {
                requiredExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
            }

            requiredExtensions.emplace_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);

            return requiredExtensions;
        }

        bool checkValidationLayerSupport() {
            uint32_t layerCount;
            vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

            std::vector<VkLayerProperties> availableLayers(layerCount);
            vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

            for (const char* layerName : VALIDATION_LAYERS) {
                bool layerFound = false;

                for (const auto& layerProperties : availableLayers) {
                    if (strcmp(layerName, layerProperties.layerName) == 0) {
                        layerFound = true;
                        break;
                    }
                }

                if (!layerFound) {
                    return false;
                }
            }

            return true;
        }

        static VkResult CreateDebugUtilsMessengerEXT(
            VkInstance instance,
            const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
            const VkAllocationCallbacks* pAllocator,
            VkDebugUtilsMessengerEXT* pDebugMessenger
        ) {
            auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
            if (func != nullptr) {
                return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
            } else {
                return VK_ERROR_EXTENSION_NOT_PRESENT;
            }
        }

        static void DestroyDebugUtilsMessengerEXT(
            VkInstance instance,
            VkDebugUtilsMessengerEXT debugMessenger,
            const VkAllocationCallbacks* pAllocator
        ) {
            auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
            if (func != nullptr) {
                func(instance, debugMessenger, pAllocator);
            }
        }

        static const std::string& messageSeverityToString(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity) {
            static const std::string MESSAGE_SEVERITY_INFO  = std::string { "INFO " };
            static const std::string MESSAGE_SEVERITY_WARN  = std::string { "WARN " };
            static const std::string MESSAGE_SEVERITY_ERROR = std::string { "ERROR" };

            if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
                return MESSAGE_SEVERITY_ERROR;
            } else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
                return MESSAGE_SEVERITY_WARN;
            } else {
                return MESSAGE_SEVERITY_INFO;
            }
        }

        static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
            VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
            VkDebugUtilsMessageTypeFlagsEXT messageType,
            const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
            void* pUserData
        ) {
            auto messageSeverityString = App::messageSeverityToString(messageSeverity);
            fmt::println(std::cerr, "[{}] {}", messageSeverityString, pCallbackData->pMessage);

            return VK_FALSE;
        }

        VkDebugUtilsMessengerCreateInfoEXT createDebugMessengerCreateInfo() {
            if (ENABLE_VALIDATION_LAYERS) {
                return VkDebugUtilsMessengerCreateInfoEXT {
                    .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
                    .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | 
                        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | 
                        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
                    .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | 
                        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | 
                        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
                    .pfnUserCallback = debugCallback,
                };
            }

            return VkDebugUtilsMessengerCreateInfoEXT {};
        }

        void createInstance() {
            if (ENABLE_VALIDATION_LAYERS && !this->checkValidationLayerSupport()) {
                throw std::runtime_error("validation layers requested, but not available!");
            }

            const auto appInfo = VkApplicationInfo {
                .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                .pApplicationName = "Hello Window",
                .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
                .pEngineName = "No Engine",
                .engineVersion = VK_MAKE_VERSION(1, 0, 0),
                .apiVersion = VK_API_VERSION_1_3,
            };
            
            const auto requiredExtensions = this->getRequiredExtensions();
            const auto flags = (VkInstanceCreateFlags {}) | VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
            
            const auto debugCreateInfo = this->createDebugMessengerCreateInfo();
            const auto debugCreateInfoPtr = [&debugCreateInfo]() -> const VkDebugUtilsMessengerCreateInfoEXT* {
                if (ENABLE_VALIDATION_LAYERS) {
                    return &debugCreateInfo;
                } else {
                    return static_cast<VkDebugUtilsMessengerCreateInfoEXT*>(nullptr);
                }
            }();

            const auto enabledLayerNames = []() -> std::vector<const char*> {
                if (ENABLE_VALIDATION_LAYERS) {
                    return VALIDATION_LAYERS;
                } else {
                    return std::vector<const char*> {};
                }
            }();

            const auto createInfo = VkInstanceCreateInfo {
                .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                .pApplicationInfo = &appInfo,
                .enabledExtensionCount = static_cast<uint32_t>(requiredExtensions.size()),
                .ppEnabledExtensionNames = requiredExtensions.data(),
                .flags = flags,
                .enabledLayerCount = static_cast<uint32_t>(enabledLayerNames.size()),
                .ppEnabledLayerNames = enabledLayerNames.data(),
                .pNext = debugCreateInfoPtr,
            };

            auto instance = VkInstance {};
            const auto result = vkCreateInstance(&createInfo, nullptr, &instance);
            if (result != VK_SUCCESS) {
                throw std::runtime_error("failed to create instance!");
            }

            m_instance = instance;
        }

        void setupDebugMessenger() {
            if (ENABLE_VALIDATION_LAYERS) {
                const auto createInfo = this->createDebugMessengerCreateInfo();

                auto debugMessenger = VkDebugUtilsMessengerEXT {};
                const auto result = App::CreateDebugUtilsMessengerEXT(m_instance, &createInfo, nullptr, &debugMessenger);
                if (result != VK_SUCCESS) {
                    throw std::runtime_error("failed to set up debug messenger!");
                }

                m_debugMessenger = debugMessenger;
            }
        }

        void createSurface() {
            auto surface = VkSurfaceKHR {};
            const auto result = glfwCreateWindowSurface(m_instance, m_window, nullptr, &surface);
            if (result != VK_SUCCESS) {
                throw std::runtime_error("failed to create window surface!");
            }

            m_surface = surface;
        }

        bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
            uint32_t extensionCount = 0;
            vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

            auto availableExtensions = std::vector<VkExtensionProperties> { extensionCount };
            vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

            auto requiredExtensions = std::set<std::string> { DEVICE_EXTENSIONS.begin(), DEVICE_EXTENSIONS.end() };

            for (const auto& extension : availableExtensions) {
                requiredExtensions.erase(extension.extensionName);
            }

            return requiredExtensions.empty();
        }

        bool isPhysicalDeviceSuitable(VkPhysicalDevice device) {
            auto indices = this->findQueueFamilies(device);

            const bool extensionsSupported = this->checkDeviceExtensionSupport(device);

            bool swapChainAdequate = false;
            if (extensionsSupported) {
                SwapChainSupportDetails swapChainSupport = this->querySwapChainSupport(device);
                swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
            }

            return indices.isComplete() && extensionsSupported && swapChainAdequate;
        }

        QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
            auto indices = QueueFamilyIndices {};

            uint32_t queueFamilyCount = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

            auto queueFamilies = std::vector<VkQueueFamilyProperties> { queueFamilyCount };
            vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

            int i = 0;
            for (const auto& queueFamily : queueFamilies) {
                if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                    indices.graphicsFamily = i;
                }

                VkBool32 presentSupport = false;
                vkGetPhysicalDeviceSurfaceSupportKHR(device, i, m_surface, &presentSupport);

                if (presentSupport) {
                    indices.presentFamily = i;
                }

                if (indices.isComplete()) {
                    break;
                }

                i++;
            }

            return indices;
        }

        void selectPhysicalDevice() {
            uint32_t physicalDeviceCount = 0;
            vkEnumeratePhysicalDevices(m_instance, &physicalDeviceCount, nullptr);

            if (physicalDeviceCount == 0) {
                throw std::runtime_error("failed to find GPUs with Vulkan support!");
            }

            auto physicalDevices = std::vector<VkPhysicalDevice> { physicalDeviceCount };
            vkEnumeratePhysicalDevices(m_instance, &physicalDeviceCount, physicalDevices.data());

            auto selectedPhysicalDevice = VkPhysicalDevice {};
            for (const auto& physicalDevice : physicalDevices) {
                if (this->isPhysicalDeviceSuitable(physicalDevice)) {
                    selectedPhysicalDevice = physicalDevice;
                    break;
                }
            }

            if (selectedPhysicalDevice == VK_NULL_HANDLE) {
                throw std::runtime_error("failed to find a suitable GPU!");
            }

            m_physicalDevice = selectedPhysicalDevice;
        }

        void createLogicalDevice() {
            const auto indices = this->findQueueFamilies(m_physicalDevice);
            const auto uniqueQueueFamilies = std::set<uint32_t> {
                indices.graphicsFamily.value(),
                indices.presentFamily.value()
            };
            
            const float queuePriority = 1.0f;
            auto queueCreateInfos = std::vector<VkDeviceQueueCreateInfo> {};
            for (uint32_t queueFamily : uniqueQueueFamilies) {
                const auto queueCreateInfo =  VkDeviceQueueCreateInfo {
                    .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                    .queueFamilyIndex = queueFamily,
                    .queueCount = 1,
                    .pQueuePriorities = &queuePriority,
                };

                queueCreateInfos.push_back(queueCreateInfo);
            }

            const auto deviceFeatures = VkPhysicalDeviceFeatures {};
            const auto enabledExtensions = []() {
                auto _enabledExtensions = std::vector<const char*> { VK_KHR_portability_subset };
                for (const char* deviceExtension : DEVICE_EXTENSIONS) {
                    _enabledExtensions.emplace_back(deviceExtension);
                }

                return _enabledExtensions;
            }();
            const auto enabledLayerNames = []() -> std::vector<const char*> {
                if (ENABLE_VALIDATION_LAYERS) {
                    return VALIDATION_LAYERS;
                } else {
                    return std::vector<const char*> {};
                }
            }();

            const auto createInfo = VkDeviceCreateInfo {
                .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                .queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size()),
                .pQueueCreateInfos = queueCreateInfos.data(),
                .pEnabledFeatures = &deviceFeatures,
                .enabledExtensionCount = static_cast<uint32_t>(enabledExtensions.size()),
                .ppEnabledExtensionNames = enabledExtensions.data(),
                .enabledLayerCount = static_cast<uint32_t>(enabledLayerNames.size()),
                .ppEnabledLayerNames = enabledLayerNames.data(),
            };

            auto device = VkDevice {};
            const auto result = vkCreateDevice(m_physicalDevice, &createInfo, nullptr, &device);
            if (result != VK_SUCCESS) {
                throw std::runtime_error("failed to create logical device!");
            }

            auto graphicsQueue = VkQueue {};
            vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
            
            auto presentQueue = VkQueue {};
            vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);

            m_device = device;
            m_graphicsQueue = graphicsQueue;
            m_presentQueue = presentQueue;
        }

        SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
            auto details = SwapChainSupportDetails {};
            vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, m_surface, &details.capabilities);

            uint32_t formatCount = 0;
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, m_surface, &formatCount, nullptr);

            if (formatCount != 0) {
                details.formats.resize(formatCount);
                vkGetPhysicalDeviceSurfaceFormatsKHR(device, m_surface, &formatCount, details.formats.data());
            }

            uint32_t presentModeCount = 0;
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, m_surface, &presentModeCount, nullptr);

            if (presentModeCount != 0) {
                details.presentModes.resize(presentModeCount);
                vkGetPhysicalDeviceSurfacePresentModesKHR(device, m_surface, &presentModeCount, details.presentModes.data());
            }

            return details;
        }

        VkSurfaceFormatKHR selectSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
            for (const auto& availableFormat : availableFormats) {
                if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                    return availableFormat;
                }
            }

            return availableFormats[0];
        }

        VkPresentModeKHR selectSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
            for (const auto& availablePresentMode : availablePresentModes) {
                if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
                    return availablePresentMode;
                }
            }

            return VK_PRESENT_MODE_FIFO_KHR;
        }

        VkExtent2D selectSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
            if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
                return capabilities.currentExtent;
            } else {
                int _width, _height;
                glfwGetWindowSize(m_window, &_width, &_height);

                const uint32_t width = std::clamp(
                    static_cast<uint32_t>(_width),
                    capabilities.minImageExtent.width,
                    capabilities.maxImageExtent.width
                );
                const uint32_t height = std::clamp(
                    static_cast<uint32_t>(_height), 
                    capabilities.minImageExtent.height, 
                    capabilities.maxImageExtent.height
                );
                const auto actualExtent = VkExtent2D {
                    .width = width,
                    .height = height,
                };

                return actualExtent;
            }
        }

        void createSwapChain() {
            const auto swapChainSupport = this->querySwapChainSupport(m_physicalDevice);
            const auto surfaceFormat = this->selectSwapSurfaceFormat(swapChainSupport.formats);
            const auto presentMode = this->selectSwapPresentMode(swapChainSupport.presentModes);
            const auto extent = this->selectSwapExtent(swapChainSupport.capabilities);
            const auto imageCount = [&swapChainSupport]() {
                uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
                if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
                    return swapChainSupport.capabilities.maxImageCount;
                }

                return imageCount;
            }();
            const auto indices = this->findQueueFamilies(m_physicalDevice);
            const auto queueFamilyIndices = std::array<uint32_t, 2> {
                indices.graphicsFamily.value(),
                indices.presentFamily.value()
            };
            const auto imageSharingMode = [&indices]() -> VkSharingMode {
                if (indices.graphicsFamily != indices.presentFamily) {
                    return VK_SHARING_MODE_CONCURRENT;
                } else {
                    return VK_SHARING_MODE_EXCLUSIVE;
                }
            }();
            const auto [queueFamilyIndicesPtr, queueFamilyIndexCount] = [&indices, &queueFamilyIndices]() -> std::tuple<const uint32_t*, uint32_t> {
                if (indices.graphicsFamily != indices.presentFamily) {
                    const auto data = queueFamilyIndices.data();
                    const auto size = static_cast<uint32_t>(queueFamilyIndices.size());
                
                    return std::make_tuple(data, size);
                } else {
                    const auto data = static_cast<uint32_t*>(nullptr);
                    const auto size = static_cast<uint32_t>(0);

                    return std::make_tuple(data, size);
                }
            }();

            const auto createInfo = VkSwapchainCreateInfoKHR {
                .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
                .surface = m_surface,
                .minImageCount = imageCount,
                .imageFormat = surfaceFormat.format,
                .imageColorSpace = surfaceFormat.colorSpace,
                .imageExtent = extent,
                .imageArrayLayers = 1,
                .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                .imageSharingMode = imageSharingMode,
                .queueFamilyIndexCount = queueFamilyIndexCount,
                .pQueueFamilyIndices = queueFamilyIndicesPtr,
                .preTransform = swapChainSupport.capabilities.currentTransform,
                .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
                .presentMode = presentMode,
                .clipped = VK_TRUE,
                .oldSwapchain = VK_NULL_HANDLE,
            };

            auto swapChain = VkSwapchainKHR {};
            const auto result = vkCreateSwapchainKHR(m_device, &createInfo, nullptr, &swapChain);
            if (result != VK_SUCCESS) {
                throw std::runtime_error("failed to create swap chain!");
            }

            uint32_t swapChainImageCount = 0;
            vkGetSwapchainImagesKHR(m_device, swapChain, &swapChainImageCount, nullptr);

            auto swapChainImages = std::vector<VkImage> { swapChainImageCount, VK_NULL_HANDLE };
            vkGetSwapchainImagesKHR(m_device, swapChain, &swapChainImageCount, swapChainImages.data());

            m_swapChain = swapChain;
            m_swapChainImages = std::move(swapChainImages);
            m_swapChainImageFormat = surfaceFormat.format;
            m_swapChainExtent = extent;
        }

        void createImageViews() {
            auto swapChainImageViews = std::vector<VkImageView> { m_swapChainImages.size(), VK_NULL_HANDLE };
            for (size_t i = 0; i < m_swapChainImages.size(); i++) {
                const auto createInfo = VkImageViewCreateInfo {
                    .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                    .image = m_swapChainImages[i],
                    .viewType = VK_IMAGE_VIEW_TYPE_2D,
                    .format = m_swapChainImageFormat,
                    .components.r = VK_COMPONENT_SWIZZLE_IDENTITY,
                    .components.g = VK_COMPONENT_SWIZZLE_IDENTITY,
                    .components.b = VK_COMPONENT_SWIZZLE_IDENTITY,
                    .components.a = VK_COMPONENT_SWIZZLE_IDENTITY,
                    .subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .subresourceRange.baseMipLevel = 0,
                    .subresourceRange.levelCount = 1,
                    .subresourceRange.baseArrayLayer = 0,
                    .subresourceRange.layerCount = 1,
                };

                auto swapChainImageView = VkImageView {};
                const auto result = vkCreateImageView(m_device, &createInfo, nullptr, &swapChainImageView);
                if (result != VK_SUCCESS) {
                    throw std::runtime_error("failed to create image views!");
                }

                swapChainImageViews[i] = swapChainImageView;
            }

            m_swapChainImageViews = std::move(swapChainImageViews);
        }

        void createWindow() {
            glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
            glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
            auto window = glfwCreateWindow(WIDTH, HEIGHT, "Hello, Window!", nullptr, nullptr);

            m_window = window;
        }

        void initVulkan() {
            this->createInstance();
            this->setupDebugMessenger();
            this->createSurface();
            this->selectPhysicalDevice();
            this->createLogicalDevice();
            this->createSwapChain();
            this->createImageViews();
        }

        void mainLoop() {
            while (!glfwWindowShouldClose(m_window)) {
                glfwPollEvents();
            }
        }

        void cleanup() {
            for (auto imageView : m_swapChainImageViews) {
                vkDestroyImageView(m_device, imageView, nullptr);
            }

            vkDestroySwapchainKHR(m_device, m_swapChain, nullptr);
            vkDestroyDevice(m_device, nullptr);

            if (ENABLE_VALIDATION_LAYERS) {
                App::DestroyDebugUtilsMessengerEXT(m_instance, m_debugMessenger, nullptr);
            }

            vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
            vkDestroyInstance(m_instance, nullptr);
            glfwDestroyWindow(m_window);
            glfwTerminate();
        }
};

int main() {
    auto app = App {};

    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
