# Problem Framing: System-Optimal Route Assignment

1. **Core Problem Definition**: This project addresses **system-optimal multi-vehicle route assignment**. It is explicitly NOT a standard single-vehicle shortest path problem (like standard Dijkstra's or A* applied to an isolated vehicle). The goal is to optimize the overall flow of multiple vehicles interacting within the network simultaneously.

2. **Computational Complexity (NP-Hardness)**: This problem is strictly NP-hard due to its combinatorial nature. If we have $V$ vehicles and each vehicle is given a set of $K$ candidate routes, the search space for a global assignment spans $K^V$ possible combinations. As the fleet size grows, an exhaustive search for the absolute best combination becomes computationally impossible.

3. **Wardrop's Principles (User Equilibrium vs. System Optimum)**: Wardrop's first principle defines User Equilibrium (UE), a state where each driver selfishly chooses the fastest available route for themselves, often leading to network-wide congestion due to shared bottlenecks. Conversely, Wardrop's second principle defines the System Optimum (SO), where routes are cooperatively assigned to minimize the *aggregate* travel time of all vehicles in the network. Achieving a System Optimum often means some vehicles must take slightly longer paths to relieve critical bottlenecks, thereby improving the efficiency of the entire system.

4. **System Guarantees and Limitations**: The system does NOT guarantee finding the global mathematical optimum, does not guarantee it will beat Dijkstra's algorithm for every single vehicle on every instance, and its congestion metrics are strictly simulated (via the BPR function) until they can be validated against real-world traffic data.
