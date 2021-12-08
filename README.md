# GLM-iprompt

Apply Iprompt on GLM(https://github.com/THUDM/GLM) with innovative new methods!

Currently support Chinese QA, English QA (Open Domain) and Chinese poem generation.

# Preparation
make sure you have installed necessary packages like Pytorch and SwissArmyTransformer (and maybe others). 
<pre>
pip install SwissArmyTransformer
</pre>


# Run
Chinese QA:
<pre>
bash generate_qa.sh config/model_glm_10B_chinese.sh
</pre>
Chinese poem:
<pre>
bash generate_pms_iprompt.sh config/model_glm_10B_chinese.sh
</pre>
English QA:
<pre>
bash generate_qa_eng.sh config/model_glm_10B.sh
</pre>

Please set up the path of the model correctly in config.

# Download Pretrained Models

Model(English): See GLM repo (https://github.com/THUDM/GLM) The 10b model gives the best performance on a single gpu. 

Model(Chinese): Please contact duzx16@mails.tsinghua.edu.cn to apply. The 10b model gives the best performance on a single gpu. 




English Sample:

<pre>
  Question: What causes the fall of Rome?
  Answer: Economic/Financial collapse and invasion by Huns.
</pre>

<pre>
  Question: What causes the fall of USA?
  Answer: 
  The cause can be seen from the following graph. It shows that USA is in fact falling down. One should notice the change in USA's position over the last few decades, since these changes ultimately lead up to its eventual collapse! USA is typically at the center in 1960, has moved to North-West in 1980 and now to South East from 2003! In case when US begins to move South-Eastwards, it will become a threat for its enemies like Iran and Russia, and hence a great source of worry to USA itself. Hence the fear is clear at this point, on which both former superpower sides of world who semblance with Iran & China are hell-bent! So it can safely be said that forces which have been against USA have now begun to push USA in one direction or another and thus leading towards its ultimate fall!With the fall of Ukraine & bankruptcy of EU-stand, things are going to get worse & worse for USA as well. America is suffering from 'Chrysler syndrome-2', where the big brands deplete and breakdown rapidly! Thanks God that good old brains behind various schemes & efforts of self destruction (such as Obamacare) did not succeed! Without them 'Socialism Of The Absurd' would have taken place in USA a long back!

Analyses revealed that positions of lesson objects in relation retain for underlines lifetime much more than for courses! Thus America will continue to hold itself together as well as NZ seems to be capable till 2040! This indicates that only gradual fall of America will occur after that period. USA will lose some market share from China and come second to Europe in terms of GDP after 2030.

</pre>

<pre>
Question: How can I keep my hair strong?

Answer: 
  Hair care is not about preventing damage, but promoting strength and maintaining your natural shape.

Here are two tips to keep your hair strong: cleanse it and condition it. Regular deep conditioning of the scalp will help lock in moisture and eliminate frizz. And I generally recommend gentle shampoos with no harsh or drying ingredients such as alcohol, sodium lauryl sulfate or oils. Conditioning regularly with a rich moisturizing treatment will soothe any rough spots, improve shine and retain strength by retaining hair's inherent moisture - around 85% is lost due to over-styling.

These tips should only be used every couple of weeks - hair needs time to rebalance itself after regular cleansing. Each one of our products can detoxify the scalp for a month, and then lubricate it again in order keep it strong. For hair that is frizzy or lacks volume or itchy scalps use Vidal Sassoon's Advanced Deep Moisturizing Shampoo. It is the only salon-formulated shampoo to retain solace between strand breakage and breakage after intense styling and chemical treatments, stops splits and breaks when applied regularly to the scalp and hair, neutralize free radical organisms responsible for frequent ageing, restore the natural glow of hair and support oily/dry hair.
</pre>

# References
GLM:https://github.com/THUDM/GLM

All NLP Tasks Are Generation Tasks: A General Pretraining Framework

Zhengxiao Du, Yujie Qian, Xiao Liu, Ming Ding, Jiezhong Qiu, Zhilin Yang, Jie Tang

iPrompt: https://github.com/THUDM/iPrompt

Controllable Generation from Pre-Trained Language Models via Inverse Prompting

Xu Zou, Da Yin, Qingyang Zhong, Hongxia Yang, Zhilin Yang, Jie Tang

