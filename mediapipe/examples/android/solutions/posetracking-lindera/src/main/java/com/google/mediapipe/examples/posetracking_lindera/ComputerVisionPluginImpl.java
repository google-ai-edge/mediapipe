package com.google.mediapipe.examples.posetracking_lindera;

import static java.lang.Math.min;

import com.google.mediapipe.solutions.lindera.BodyJoints;
import com.google.mediapipe.solutions.lindera.ComputerVisionPlugin;
import com.google.mediapipe.solutions.lindera.XYZPointWithConfidence;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.lang.reflect.Field;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Locale;
import java.util.Map;

public class ComputerVisionPluginImpl implements ComputerVisionPlugin {

    LinkedList<BodyJointsEvent> bodyJointsEventList  = new LinkedList<>();
    static class BodyJointsEvent{
        BodyJoints bodyJoints;
        Long timestamp;

        public BodyJointsEvent(long timestamp, BodyJoints bodyJoints) {
            this.bodyJoints = bodyJoints;
            this.timestamp  = timestamp;
        }
    }




    boolean isLogging = false;

    public void startLogging(){
        isLogging = true;
        bodyJointsEventList  = new LinkedList<>();
    }

    public JSONObject stopLoggingAndDumpOutput() throws JSONException, IllegalAccessException {
        isLogging = false;
        // base json string
        String json = "{\n" +
                "  \"identifier\": \"some_name_here.Capture\",\n" +
                "  \"metaData\": {\n" +
                "    \"userIdentifier\": \"unknown user\",\n" +
                "    \"originatingEquipment\": \"Apperture\",\n" +
                "    \"activityName\": \"exercise name\",\n" +
                "    \"activityDescription\": \"\",\n" +
                "    \"tags\": [\n" +
                "      \n" +
                "    ]\n" +
                "  },\n" +
                "  \"bodyObjects\": [\n" +
                "    \n" +
                "  ]}";

        JSONObject eLog = new JSONObject(json);
        JSONArray bodyJointsArr = new JSONArray();
        for (BodyJointsEvent bodyJointsEvent:bodyJointsEventList){
            JSONObject jBodyJointEvent = new JSONObject();
            jBodyJointEvent.put("ts",bodyJointsEvent.timestamp);
            BodyJoints bodyJoints = bodyJointsEvent.bodyJoints;
            String bodyJointsString = "";
            // iterate over fields of BodyJoints and put them in json format
            for (Field field : BodyJoints.class.getDeclaredFields()) {
                Class<?> type = field.getType();

                if (type ==  XYZPointWithConfidence.class) {

                    String name = field.getName();
                    // get abbreviation of name for example leftShoulder -> LS
                    String abbrev = String.valueOf(name.charAt(0));
                    for (int i = 1;i<name.length();++i){
                        char chari = name.charAt(i);
                        if (Character.isUpperCase(chari)){
                           abbrev =  abbrev.concat(String.valueOf(chari));
                        }
                    }
                    abbrev = abbrev.toUpperCase(Locale.ROOT);
                    XYZPointWithConfidence data = (XYZPointWithConfidence) field.get(bodyJoints);
                    assert data != null;
                    bodyJointsString = bodyJointsString.concat(String.format(abbrev+":%f,%f,%f=",data.x,data.y,data.z));



                }
            }
            bodyJointsString = bodyJointsString.concat(interpolateJoints(bodyJoints));
            // remove the last equal sign
            bodyJointsString = bodyJointsString.substring(0,bodyJointsString.length()-1);

            jBodyJointEvent.put("bj",bodyJointsString);

            bodyJointsArr.put(jBodyJointEvent);



        }
        eLog.put("bodyJoints",bodyJointsArr);

        return eLog;


    }

    String interpolateJoints(BodyJoints bodyJoints){

        Map<String,XYZPointWithConfidence> pts = new HashMap<>();
        pts.put("PE", getPelvis(bodyJoints));
        pts.put("NN",bodyJoints.nose);
        // Assuming Thorax is 1/3 of distance between shoulders and pelvis
        XYZPointWithConfidence thorax = getSpinePoint(bodyJoints,1/3f);
        // Assuming spine/middle back is 2/3 of distance between shoulders and pelvis
        XYZPointWithConfidence spine = getSpinePoint(bodyJoints,2/3f);
        pts.put("TH",thorax);
        pts.put("SP",spine);
        pts.put("HT",getHeadTop(bodyJoints));
        final String[] bodyJointsString = {""};
        pts.forEach((key,data)->{
            bodyJointsString[0] = bodyJointsString[0].concat(String.format(key+":%f,%f,%f=",data.x,data.y,data.z));

        });

        return bodyJointsString[0];





    }
    XYZPointWithConfidence getPelvis(BodyJoints bodyJoints){
        return getMiddleJoint(bodyJoints.leftHip,bodyJoints.rightHip);

    }

    XYZPointWithConfidence getJointBetweenPoints(XYZPointWithConfidence pt1,XYZPointWithConfidence pt2,float distance){
        XYZPointWithConfidence midpt = new XYZPointWithConfidence();
        midpt.x  = pt1.x + (pt2.x-pt1.x)*distance;
        midpt.y  = pt1.y + (pt2.y-pt1.y)*distance;
        midpt.z  = pt1.z + (pt2.z-pt1.z)*distance;


        midpt.presence = min(pt1.presence,pt2.presence);
        midpt.confidence = min(pt1.confidence,pt2.confidence);
        return midpt;
    }
    XYZPointWithConfidence getMiddleJoint(XYZPointWithConfidence pt1,XYZPointWithConfidence pt2) {
        XYZPointWithConfidence midpt = new XYZPointWithConfidence();
        midpt.x  = (pt1.x + pt2.x)/2;
        midpt.y  = (pt1.y + pt2.y)/2;
        midpt.z  = (pt1.z + pt2.z)/2;

        midpt.presence = min(pt1.presence,pt2.presence);
        midpt.confidence = min(pt1.confidence,pt2.confidence);
        return midpt;
    }
        XYZPointWithConfidence getSpinePoint(BodyJoints bodyJoints, float distanceFromShoulders){
            XYZPointWithConfidence midShoulder = getMiddleJoint(bodyJoints.leftShoulder, bodyJoints.rightShoulder);
            XYZPointWithConfidence pelvis = getPelvis(bodyJoints);
            return getJointBetweenPoints(midShoulder,pelvis,distanceFromShoulders);
    }

    XYZPointWithConfidence getHeadTop(BodyJoints bodyJoints){
        XYZPointWithConfidence middleEye = getMiddleJoint(bodyJoints.leftEye,bodyJoints.rightEye);
        return getJointBetweenPoints(middleEye,bodyJoints.nose,2);
    }



    @Override
    public void bodyJoints(long timestamp, BodyJoints bodyJoints) {
        if (isLogging){
            this.bodyJointsEventList.add(new BodyJointsEvent(timestamp,bodyJoints));

        }
//        XYZPointWithConfidence nose = bodyJoints.nose;
//        Log.v("ComputerVisionPluginImpl", String.format(
//
//                "Lindera BodyJoint of Nose: x=%f, y=%f, z=%f", nose.x, nose.y, nose.z));


    }
}
