<div align="center">
  <div>
    <a href="https://strandsagents.com">
      <img src="https://strandsagents.com/latest/assets/logo-github.svg" alt="Strands Agents" width="55px" height="105px">
    </a>
  </div>
</div>

# Strands Benchmark Harnesses

A repository for Strands-based agents and harnesses for agentic benchmarks. It is a
[uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/): the repository
root coordinates one or more member packages. Setup, configuration, and usage live in each agent's README.

## Agents

### [SSA (Simple Strands Agent)](simple-strands-agent/README.md)

A minimal, hackable agent harness achieving state-of-the-art performance across software engineering benchmarks (SWE-Bench Verified, SWE-Bench Pro, Terminal-Bench-2.0).

See [simple-strands-agent/README.md](simple-strands-agent/README.md) for setup, configuration, and usage.

## Running agents safely

Agents in this repository are given access to shell tools. In practice, this means the model can run commands in the environment where the agent is started.

This is useful for experiments and benchmarking, but it also means you should treat the agent like you would treat any program with shell access: it may read files, modify files, delete data, install packages, or accidentally expose information from the environment.

For normal use, we recommend running agents in an isolated environment rather than directly on your machine. Our experiments and benchmarks are run inside Docker containers. You should avoid running agents in an environment that contains secrets, credentials, personal files, production data, or anything you would not want the model to access.

A good default setup is:

- run the agent inside Docker or another sandboxed environment
- mount only the files/directories the agent actually needs
- avoid exposing cloud credentials, SSH keys, API keys, or other secrets
- do not run it with unnecessary privileges
- inspect outputs before trusting or reusing them

Agents will usually behave as instructed, but shell access is powerful. Use the same caution you would use when running code from an automated system.

## License

[Apache 2.0](LICENSE)
