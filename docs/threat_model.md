# Threat model

**Protects against:** indirect prompt injection that arrives through tool outputs (emails, web pages, files,
API responses, MCP tool results) and steers an agent into tool calls the user did not ask for.

**How:** before each proposed call executes, TraceWarden scores it using the user's task, the user's known
world (own address, contacts, domains), and everything the agent has seen so far. Calls above calibrated
thresholds are held for review or blocked.

**Does not protect against:**
* direct jailbreaks typed by the user, or a malicious user;
* attacks that achieve their goal purely in the agent's final text answer (no tool call to intercept);
* a compromised tool or MCP server that lies about what it did;
* adaptive attackers who study the detector; expect evasion to be possible (see Day 18 results);
* attacks far from the training distribution (new tools, languages, formats) until you adapt and recalibrate.

**Operating guidance:**
* treat it as one layer: least-privilege tools, confirmation for high-impact actions, input sanitization;
* recalibrate thresholds on a sample of your own benign traffic (`tracewarden calibrate`), because the conformal
  guarantee only holds for data like the calibration set;
* `taint_policy="hold"` adds a deterministic rule for out-of-distribution attacks at a real false-hold cost
  (22.5% of benign AgentDrift test runs); prefer it where a human is in the loop anyway;
* log decisions; the rollback report assumes the log is complete.
