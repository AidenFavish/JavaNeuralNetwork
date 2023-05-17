package com.aidenfavish.javaNeuralNetwork.Optimizers;

import com.aidenfavish.javaNeuralNetwork.Layers.LayerPass;
import com.aidenfavish.javaNeuralNetwork.Models.Model;
import com.aidenfavish.javaNeuralNetwork.Resources.Environment;
import com.aidenfavish.javaNeuralNetwork.Resources.Matrix2D;

import java.util.ArrayList;
import java.util.HashMap;

public class PPO {
    private int timestepsPerBatch;
    private int maxTimestepsPerEpisode;
    private int nUpdatesPerIteration;
    private float gamma;
    private float clip;
    private int saveFreq; // How often we save in number of iterations

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
    }

    public void learn(int totalTimesteps) {
        System.out.println("Learning... Running " + maxTimestepsPerEpisode + " timesteps per episode");
        System.out.println(timestepsPerBatch + " timesteps per batch for a total of " + totalTimesteps + "timesteps");
        int tSoFar = 0;
        int iSoFar = 0;

        while (tSoFar < totalTimesteps) {
            // TODO Collect batch simulations here
            //  batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()


        }
    }
}
