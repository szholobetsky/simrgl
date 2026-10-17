# Tasks, group A

## A1

Fix documentation about synchronization of default group when using SSO authentication
In sonar.properties, the documentation of synchronization of groups when using SSO authentication is wrong, as it's written that the default group is not synchronized which is wrong since 6.4 ( https://jira.sonarsource.com/browse/SONAR-9014 ). Current documentation is :  # Name of the header to get the list of user groups, separated by comma (optional).
# If the sonar.sso.groupsHeader is set, the user will belong to those groups if groups exist in SonarQube.
# If none of the provided groups exists in SonarQube, the user won't belong to any group.
# Note that the default group will NOT be automatically added when using SSO, it should be provided in the groups list, if needed.
#sonar.web.sso.groupsHeader=X-Forwarded-Groups It should become :  # Name of the header to get the list of user groups, separated by comma (optional).
# If the sonar.sso.groupsHeader is set, the user will belong to those groups if groups exist in SonarQube.
# If none of the provided groups exists in SonarQube, the user will only belong to the default group.
# Note that the default group will always be set.
#sonar.web.sso.groupsHeader=X-Forwarded-Groups

## A2

Do not display FOP INFO logs when generating a PDF report
FOP displays logs each time a report is generated, for each page. The FOP logger should be overloaded with something similar to in  FileReportGenerator : Logger logger = Loggers.get("org.apache.fop");
logger.setLevel(LoggerLevel.ERROR);

## A3

Project creation form validation is buggy
When creating a project, the field validation kicks in for wrong/unexpected formats, but doesn't actually prevent creation. Also, some formats that should be working (according to description) aren't accepted.

## A4

Projects can be sorted by Security Review Rating
In the Projects page, it should be possible to sort Projects by Security Review Rating. This metric was introduced when we implements the Security Hotspots page/experience and we forgot to update the Projects page accordingly.

## A5

Remove no more used Sonar-Version HTTP header
The  Sonar-Version  HTTP header was used by the blue/green feature, which is no more used on SonarQube, and which has never been a public feature. As it's no more used anymore, this header should be removed.
