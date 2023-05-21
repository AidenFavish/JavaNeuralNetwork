package com.aidenfavish.javaNeuralNetwork.Optimizers;

import com.aidenfavish.javaNeuralNetwork.Layers.LayerPass;
import com.aidenfavish.javaNeuralNetwork.Models.Model;
import com.aidenfavish.javaNeuralNetwork.Resources.Environment;
import com.aidenfavish.javaNeuralNetwork.Resources.Matrix2D;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.HashMap;

public class PPO {
    // Hyperparameters
    private int timestepsPerBatch;
    private int maxTimestepsPerEpisode;
    private int nUpdatesPerIteration;
    private float gamma;
    private float clip;
    private int saveFreq; // How often we save in number of iterations
    private int renderEveryI;

    private Environment env;
    private int obsDim;
    private int actDim;
    private Model actor;
    private Model critic;
    private Matrix2D covVar;
    private Matrix2D covMat;
    private HashMap<String, Object> logger;

    public PPO(Model actorClass, Model criticClass, Environment env, HashMap<String, Object> hyperparameters) {
        this.actor = actorClass;
        this.critic = criticClass;
        this.env = env;
        this.obsDim = env.getObsSpaceShape();
        this.actDim = env.getActSpaceShape();

        // TODO initialize the covariance variables used to query the actor
        //  covVar =
        //  covMat =

        logger = new HashMap<>();
        logger.put("delta t", System.nanoTime());
        logger.put("t so far", 0); // Timesteps so far
        logger.put("i so far", 0); // Iterations so far
        logger.put("batch lens", new ArrayList<Integer>());
        logger.put("batch rews", new ArrayList<Integer>());
        logger.put("batch losses", new ArrayList<Float>());

        getHyperparameters(hyperparameters);
    }

    public void learn(int totalTimesteps) {
        System.out.println("Learning... Running " + maxTimestepsPerEpisode + " timesteps per episode");
        System.out.println(timestepsPerBatch + " timesteps per batch for a total of " + totalTimesteps + "timesteps");
        int tSoFar = 0;
        int iSoFar = 0;

        ArrayList<float[]> batchObs;
        ArrayList<Integer> batchActs;
        ArrayList<Float> batchLogProbs;
        ArrayList<Float> batchRtgs;
        ArrayList<Integer> batchLens;
        HashMap<String, Object> tempHashMap;
        Object[] temp;
        float[] V;
        float[] Ak;
        float[] currLogProb;
        float[] ratios;
        float[] surr1;
        float[] surr2;
        float actorLoss;
        float criticLoss;

        while (tSoFar < totalTimesteps) {
            tempHashMap = rollout();
            batchObs = (ArrayList<float[]>) tempHashMap.get("batch obs");
            batchActs = (ArrayList<Integer>) tempHashMap.get("batch acts");
            batchLogProbs = (ArrayList<Float>) tempHashMap.get("batch log probs");
            batchRtgs = (ArrayList<Float>) tempHashMap.get("batch rtgs");
            batchLens = (ArrayList<Integer>) tempHashMap.get("batch lens");

            for (Integer i: batchLens) {
                tSoFar += i;
            }

            iSoFar += 1;

            logger.replace("t so far", tSoFar);
            logger.replace("i so far", iSoFar);

            temp = evaluate(batchObs, batchActs);
            V = (float[]) temp[0];
            Ak = new float[V.length];
            for (int i = 0; i < V.length; i++) {
                Ak[i] = batchRtgs.get(i) - V[i];
            }

            float mean = calculateMean(Ak);
            float std = calculateSD(Ak);
            for (int i = 0; i < Ak.length; i++) {
                Ak[i] = (Ak[i] - mean) / (std + (float)Math.pow(10, -10));
            }

            for (int i = 0; i < nUpdatesPerIteration; i++) {
                temp = evaluate(batchObs, batchActs);
                V = (float[]) temp[0];
                currLogProb = (float[]) temp[1];

                ratios = calculateRatios(currLogProb, batchLogProbs);

                surr1 = new float[ratios.length];
                for (int j = 0; j < ratios.length; j++) {
                    surr1[j] = ratios[j] * Ak[j];
                }
                surr2 = calculateSurr2(ratios, Ak);

                actorLoss = calculateActorLoss(surr1, surr2);
                criticLoss = calculateCriticLoss(V, batchRtgs);
            }
        }
    }

    private float calculateCriticLoss(float[] v, ArrayList<Float> batchRtgs) {
        float[] ans = new float[v.length];
        for (int i = 0; i < v.length; i++) {
            ans[i] = (float)Math.pow(batchRtgs.get(i) - v[i], 2);
        }
        return calculateMean(ans);
    }

    private float calculateActorLoss(float[] surr1, float[] surr2) {
        float[] ans = new float[surr1.length];
        for (int i = 0; i < surr1.length; i++) {
            ans[i] = -Math.min(surr1[i], surr2[i]);
        }
        return calculateMean(ans);
    }

    private float[] calculateSurr2(float[] ratios, float[] ak) {
        float[] ans = new float[ratios.length];
        for (int i = 0; i < ratios.length; i++) {
            if (ratios[i] > 1 + clip) {
                ans[i] = (1 + clip) * ak[i];
            } else if (ratios[i] < 1 - clip) {
                ans[i] = (1 - clip) * ak[i];
            } else {
                ans[i] = ratios[i] * ak[i];
            }
        }
        return ans;
    }

    private float[] calculateRatios(float[] currLogProb, ArrayList<Float> batchLogProbs) {
        float[] ans = new float[currLogProb.length];
        for (int i = 0; i < currLogProb.length; i++) {
            ans[i] = (float)Math.exp(currLogProb[i] - batchLogProbs.get(i));
        }
        return ans;
    }

    public HashMap<String, Object> rollout() {
        HashMap<String, Object> ans = new HashMap<>();
        ArrayList<float[]> batchObs = new ArrayList<>();
        ArrayList<Integer> batchActs = new ArrayList<>();
        ArrayList<Float> batchLogProb = new ArrayList<>();
        ArrayList<ArrayList<Float>> batchRews = new ArrayList<>();
        ArrayList<Float> batchRtgs = new ArrayList<>();
        ArrayList<Integer> batchLens = new ArrayList<>();
        Object[] temp;
        int action;
        float logProb;
        boolean done;
        float rew;

        ArrayList<Float> epRews = new ArrayList<>();
        int t = 0; // Keeps track of how many timestamps we have run so far
        float[] obs;
        int epT = 0;

        while (t < timestepsPerBatch) {
            epRews = new ArrayList<>(); // rewards collected per episode
            obs = env.reset();
            env.setDone(false);

            for (epT = 0; epT < maxTimestepsPerEpisode; epT++) {
                if ((Integer)logger.get("i so far") % renderEveryI == 0 && batchLens.size() == 0) {
                    env.render();
                }

                t += 1; // increment timesteps for this batch

                batchObs.add(obs);

                // Calculate action and ake a step in the env
                temp = getAction(obs);
                action = (Integer)temp[0];
                logProb = (Float)temp[1];

                temp = env.step(action);
                obs = (float[])temp[0];
                rew = (float)temp[1];
                done = (boolean)temp[2];

                // Track reward, action, and action log prob
                epRews.add(rew);
                batchActs.add(action);
                batchLogProb.add(logProb);

                if (done)
                    break;
            }
            // Track episode lengths and rewards
            batchLens.add(epT + 1);
            batchRews.add(epRews);
        }
        // Computer rewards to go
        batchRtgs = computeRtgs(batchRews);

        logger.replace("batch rews", batchRews);
        logger.replace("batch lens", batchLens);

        // Build the return
        ans.put("batch obs", batchObs);
        ans.put("batch acts", batchActs);
        ans.put("batch log probs", batchLogProb);
        ans.put("batch rtgs", batchRtgs);
        ans.put("batch lens", batchLens);

        return ans;
    }

    private void getHyperparameters(HashMap<String, Object> set) {
        if (set.containsKey("timesteps per batch")) {
            timestepsPerBatch = (Integer)set.get("timesteps per batch");
        } else {
            timestepsPerBatch = 4800;
        }
        if (set.containsKey("max timesteps per episode")) {
            maxTimestepsPerEpisode = (Integer)set.get("max timesteps per episode");
        } else {
            maxTimestepsPerEpisode = 1600;
        }
        if (set.containsKey("n updates per iteration")) {
            nUpdatesPerIteration = (Integer)set.get("n updates per iteration");
        } else {
            nUpdatesPerIteration = 5;
        }
        if (set.containsKey("gamma")) {
            gamma = (Float)set.get("gamma");
        } else {
            gamma = 0.95f;
        }
        if (set.containsKey("clip")) {
            clip = (Float)set.get("clip");
        } else {
            clip = 0.2f;
        }
        if (set.containsKey("render every i")) {
            renderEveryI = (Integer)set.get("render every i");
        } else {
            renderEveryI = 10;
        }
    }

    public Object[] getAction(float[] obs) {
        int action = Model.sample(actor.getProb(obs));
        Object[] ans = new Object[2];
        ans[0] = action;
        ans[1] = Math.log(actor.getProb(obs)[action]);
        return ans;
    }

    private ArrayList<Float> computeRtgs(ArrayList<ArrayList<Float>> batchRews) {
        ArrayList<Float> batchRtgs = new ArrayList<>();
        float discountReward = 0;
        for (int epRews = batchRews.size() - 1; epRews >= 0; epRews--) {
            discountReward = 0;
            for (int rew = batchRews.get(epRews).size() - 1; rew >= 0; rew--) {
                discountReward = rew + discountReward * gamma;
                batchRtgs.add(0, discountReward);
            }
        }
        return batchRtgs;
    }

    private Object[] evaluate(ArrayList<float[]> batchObs, ArrayList<Integer> batchActs) {
        float[] V = new float[batchObs.size()];
        for (int i = 0; i < batchObs.size(); i++) {
            V[i] = critic.predictDisct(batchObs.get(i));
        }

        float[] logProbs = new float[batchObs.size()];

        for (int i = 0; i < batchActs.size(); i++) {
            logProbs[i] = actor.getProb(batchObs.get(i))[batchActs.get(i)];
        }

        Object[] ans = new Object[2];
        ans[0] = V;
        ans[1] = logProbs;
        return ans;
    }

    private float calculateSD(float[] numArray)
    {
        float sum = 0.0f, standardDeviation = 0.0f;
        int length = numArray.length;

        for(float num : numArray) {
            sum += num;
        }

        float mean = sum/length;

        for(float num: numArray) {
            standardDeviation += (float)Math.pow(num - mean, 2);
        }

        return (float)Math.sqrt(standardDeviation/length);
    }

    private float calculateMean(float[] numArray) {
        float sum = 0;
        for (float f: numArray) {
            sum += f;
        }
        return sum / numArray.length;
    }
}
