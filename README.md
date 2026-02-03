# Test bed 

Lunar lander with two goals, one on the left and one on the right.

<table>
  <tr>
    <td><img src="gif/heuristic_left_ep0.gif" width="180"><br>Left Expert</td>
    <td><img src="gif/heuristic_left_ep1.gif" width="180"><br>Left Expert</td>
    <td><img src="gif/heuristic_left_ep2.gif" width="180"><br>Left Expert</td>
    <td><img src="gif/heuristic_left_ep3.gif" width="180"><br>Left Expert</td>
  </tr>
  <tr>
    <td><img src="gif/heuristic_right_ep0.gif" width="180"><br>Right Expert</td>
    <td><img src="gif/heuristic_right_ep1.gif" width="180"><br>Right Expert</td>
    <td><img src="gif/heuristic_right_ep2.gif" width="180"><br>Right Expert</td>
    <td><img src="gif/heuristic_right_ep3.gif" width="180"><br>Right Expert</td>
  </tr>
</table>


Agent 1: A very dumb VLM trying to solve the task:

<table>
  <tr>
    <td><img src="gif/episode_0_goal+0.200.gif" width="180"><br>Right Student</td>
    <td><img src="gif/episode_1_goal+0.200.gif" width="180"><br>Right Student</td>
    <td><img src="gif/episode_2_goal-0.200.gif" width="180"><br>Left Student</td>
    <td><img src="gif/episode_3_goal-0.200.gif" width="180"><br>Left Student</td>
  </tr>
</table>

We can see that they sometimes can't even distinguish left and right:

```
{"type": "step", "episode": 0, "t": 0, "goal_x": 0.20000000298023224, "goal_side": "RIGHT", "action": [0.10000000149011612, -0.5], "reward": -0.551408953038947, "terminated": false, "truncated": false, "goal_dx": -0.2054570228792727, "goal_success": false, "vlm_called": true, "parsed_ok": true, "raw_vlm": "<think>Got it, let's look at the problem. The goal_x is 0.200, which is positive, so we need to go RIGHT. For lateral movement, moving right means using lateral <= -0.5 (left booster). Let's check the current position. The pad has two flags, and the goal_x is positive, so we need to move right. So lateral should be a negative value, maybe -0.5. The main engine might need to adjust, but since we're focusing on lateral first, let's set main maybe to 0.1 or so, but wait, the task is to land, so maybe start with main engine off or low. Wait, the action is [main, lateral]. Let's think: to move right, lateral should be <= -0.5, so let's use lateral = -0.5. Main engine maybe 0.1? Wait, but let's check the image. The lander is above, and the pad is under. To move right, use left booster (lateral <= -0.5). So action would be [main, lateral] = [0.1, -0.5] or similar. Wait, the problem says \"land on the pad near goal_x\". Since goal_x is positive, go right. So lateral should be negative (left booster). Let's set main to maybe 0.1 (thrust a bit) and lateral to -0.5. So the action is [0.1, -0.5]. Wait, but need to make sure. Alternatively, maybe main is 0, but no, need to control. Wait, the initial step, maybe start with main engine off (main=0) and lateral=-0.5. Wait, let's check the rules again. For lateral: lateral <= -0.5 is left booster (move right). So action is [main, lateral] = [0.0, -0.5] or [0.1, -0.5]. Let's pick main as 0.1, lateral as -0.5. So the JSON would be {\"action\":[0.1, -0.5]}."}
```
But they know that they need to go right, and they also use the right action scale: [0.10000000149011612, -0.5] means the left engine is working so the robot should go right. However there is a gap between the thoughts, action, and what really happens.


I have a simple automaton toy, which the expert will not have knowledge of where the goal is for the student and only infer the students intension through the action and thoughs and offer 60% of weight helping: we use vlm for every 5 steps, and the vlm will output an action and thoughs. Then the expert will guess the intension of the vlm and offer help.



<table>
  <tr>
    <td><img src="gif/ep0.gif" width="180"><br>Left Hybrid</td>
    <td><img src="gif/ep4.gif" width="180"><br>Left Hybrid</td>
    <td><img src="gif/ep2.gif" width="180"><br>Right Hybrid</td>
    <td><img src="gif/ep3.gif" width="180"><br>Right Hybrid</td>
  </tr>
</table>


