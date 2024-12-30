# scaling law

小模型没有大模型的涌现现象，同时训练的表现也很不一样

训练的时候用了很多数据，而这些数据里面就有很多错误的/自相矛盾的内容。所以直接用base model来回答会有很多错误的结果，这也是为什么需要做 alignment 的原因（RLHF）

RLHF不会提升模型的能力，只是控制模型的输出方向，让模型更知道我们的意图

可以用 system message 来引导模型输出内容的风格了

仍然会有幻觉 hallucinates fact 和 reasoning error