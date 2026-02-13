import math

from util import _extract_number


def rule_reward(prompts, completions, completion_ids, **reward_kwargs):
    data_source, ability, reward_model, extra_info, trainer_state = reward_kwargs.values()
    rewards = []
    for i, completion in enumerate(completions):
        gt_num = _extract_number(reward_model[i]["ground_truth"])
        completion_num = _extract_number(completion[0]["content"])
        if completion_num == gt_num:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def get_ndcg_rule_reward(num_beams):
    ndcg_rewards = [-1.0 / math.log2(i + 2) for i in range(num_beams)]
    ndcg_rewards = [-elm / sum(ndcg_rewards) for elm in ndcg_rewards]

    def ndcg_rule_reward(prompts, completions, completion_ids, **reward_kwargs):
        data_source, ability, reward_model, extra_info, trainer_state = reward_kwargs.values()
        repeat = num_beams
        rewards = []
        flag = False
        lis = []

        for i, completion in enumerate(completions):
            completion_num = _extract_number(completion[0]["content"])
            gt_num = _extract_number(reward_model[i]["ground_truth"])
            if completion_num == gt_num:
                flag = True
                lis.append(0.0)
            else:
                lis.append(ndcg_rewards[i % num_beams])
            if (i + 1) % num_beams == 0:
                if flag:
                    rewards.extend(lis)
                else:
                    rewards.extend([0.0] * repeat)
                flag = False
                lis = []
        return rewards

    return ndcg_rule_reward
