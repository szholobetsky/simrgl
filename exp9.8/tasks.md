# Tasks D1..D5

## D1

Correctly display changelog of web service past updates
When the response of a web service is updated, for the moment we add a line in the description of the action but it's not readable and not enough visible. A changelog feature should be introduced to understand past changes that happen in the web service.

## D2

Restrict scope of api/issues/tags 
Add optional  project  parameter This web service should not search in rules tags to find a match

## D3

Create an option panel on top of projects page
Move the view option in the panel Move the graph options inside the same select as the view option Do panel animations

## D4

Telemetry Improvements
WHY We observe some inconsistencies in telemetry data. In order to clean up the data, we need to improve the payload in order to better understand what is happening. WHAT Make adjustments to the telemetry payload to further distinguish messages from each other. Current problems: It seems that the telemetry ping is sent again after a SonarQube upgrade, or a database upgrade  => this could not be reproduced Multiple telemetry pings are sent on the same day, when we expect only one a day HOW Investigate whether we should do a bug fix to not send a 2nd message after an SQ upgrade or downgrade Information about the validity of the license (to help understand if the ping might come from a restored backup on another database: the license would be invalid in this case) Timestamp of the local machine (to help understand what’s happening on SQ side when multiple pings are sent every day) Send a sequence (stored in the database) starting at 1, and increasing with every ping sent. This sequence is never reset. This will hopefully help us understand when the ping is from “the past” (A backup was restored), or if several instances are running with the same server id.

## D5

Deploy core plugins to maven repository
This is required, because now we can have plugin inter-dependencies (see  ).
