package ca.copperlabs.mediapipe.examples.posetracking_lindera;

import static java.lang.Math.min;

import ca.copperlabs.cv.BodyJoints;
import ca.copperlabs.cv.ComputerVisionPlugin;
import ca.copperlabs.cv.XYZPointWithConfidence;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.lang.reflect.Field;
import java.util.LinkedList;
import java.util.Locale;

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
                    // correct abbreviations here
                    switch (abbrev) {
                        case "P":
                            abbrev = "PE";
                            break;
                        case "T":
                            abbrev = "TH";
                            break;
                        case "S":
                            abbrev = "SP";
                            break;
                    }
                    XYZPointWithConfidence data = (XYZPointWithConfidence) field.get(bodyJoints);
                    assert data != null;
                    bodyJointsString = bodyJointsString.concat(String.format(abbrev+":%f,%f,%f=",data.x,data.y,data.z));



                }
            }
            // remove the last equal sign
            bodyJointsString = bodyJointsString.substring(0,bodyJointsString.length()-1);

            jBodyJointEvent.put("bj",bodyJointsString);

            bodyJointsArr.put(jBodyJointEvent);



        }
        eLog.put("bodyJoints",bodyJointsArr);

        return eLog;


    }








    @Override
    public void bodyJoints(Long timestamp, BodyJoints bodyJoints) {
        if (isLogging){
            this.bodyJointsEventList.add(new BodyJointsEvent(timestamp,bodyJoints));

        }
//        XYZPointWithConfidence nose = bodyJoints.nose;
//        Log.v("ComputerVisionPluginImpl", String.format(
//
//                "Lindera BodyJoint of Nose: x=%f, y=%f, z=%f", nose.x, nose.y, nose.z));


    }
}
