# Etch to VLA Forge: what the widget does and what was ported

Source: Etch, the author's own earlier project (https://github.com/PranavAchar01/etch, deploy/etch-app at a0bb241d).
Files read: js/ui/{dock,eyes,goo,spring,trace,voice-adapter}.js, js/voice.js, js/mascot.js, js/editor/agent-bridge.js,
css/agent.css. Code was copied and then adapted only in selectors, colours, strings and events. Edits are marked "Meta:"
or sit in the header comment of each file.

One fact up front: Etch's character does not roam. It lives in the dock pill and only looks at the part being worked on
(`dock.lookAt`). Roaming the whole screen is new in the forge (`web/roam.js`), built on Etch's own spring presets,
wobble, squash and emotes.

| # | Etch feature | Etch source | Forge file | Status |
|---|---|---|---|---|
| 1 | Eye rig: states, activities, emotes, blink and glance timers, gaze maths | js/ui/eyes.js | web/eyes.js | done (round 8, Meta glasses and antenna added) |
| 2 | Spring physics and presets | js/ui/spring.js | web/spring.js | done (round 8) |
| 3 | Gooey bubble layer | js/ui/goo.js | web/goo.js | done (round 8) |
| 4 | Page events to states, emotes, bubbles | css/agent.css bubbles, js/mascot.js idea | web/mascot.js | done (round 8), extended in round 9 (roamer, mood event, bubble follow) |
| 5 | Dock: pill, liquid open and close, composer, mic, status line, turn timeline, fold of finished work, unseen-count pill | js/ui/dock.js | web/dock.js, web/dock.css | done, with edits: pill at the bottom right, the roaming bot sits in its eye slot, bubbles drawn by mascot.js |
| 6 | `dock.step` rows: step, reasoning, tool, task with running, done, failed | js/ui/dock.js, trace.js | web/forge-agent.js | done: every agent thought is a step row, Llama ones are reasoning rows |
| 7 | Thinking trace: the agent's reasoning shown as it goes | js/ui/trace.js `thinkingTrace` | web/trace.js, plus the Thinking panel in web/forge-agent.js | done; the panel is new and shows each thought with its agent and whether Llama answered |
| 8 | Streamed answer with bold, code, mention chips and follow-ups | js/ui/trace.js `streamAnswer` | web/trace.js, web/forge-agent.js | done: mentions are agent names, hovering or clicking one flies the bot to that agent's area |
| 9 | "What are you doing?" empty state and answer | dock empty state, agent-bridge answers | web/dock.js, web/forge-agent.js | done: clicking the bot or typing it answers from the live stage and the last agent thoughts |
| 10 | Loader with elapsed seconds | js/editor/agent-bridge.js `loader` | web/forge-agent.js | done |
| 11 | Look at the thing being worked on (`dock.lookAt`, highlight) | agent-bridge.js | web/roam.js, web/mascot.js | done and extended: the bot flies to it and points with its arm |
| 12 | Approval cards (`ask`) and recommend cards | trace.js `approvalCard`, `recommendCard`, agent-bridge.js | web/trace.js (code kept) | not wired: the forge never changes the user's work without being asked, so there is nothing to approve |
| 13 | Voice: push to talk (hold Space), mic level wave, captions | js/voice.js, js/ui/voice-adapter.js | web/voice-adapter.js | done with the browser's SpeechRecognition; Etch's engine needs Etch's backend. It does not speak out loud |
| 14 | Queue state ("Waiting in line") | agent-bridge.js `lead:queued` | web/dock.js act table | partial: the label is there, the forge has no queue events yet |
| 15 | Bench / benchmark card | js/ui/bench-panel.js | none | not ported: it measures PCB boards |
| 16 | Board-specific act table (route, place, repair) | dock.js ACTS | web/dock.js | rewritten for the forge stages |
| 17 | Reduced motion handling | spring.js, dock.js | all | done |
| 18 | Pixel loader and icons | trace.js | web/trace.js | done |

New in the forge, not in Etch: the roaming robot body with a gripper arm, visor seam, side pods and hover jets
(web/roam.js, roam.css); the Thinking panel (key T); the Agents view with live message pulses (web/agents.js).
