# Tasks T01..T10

## T01

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

## T02

Change message in projects list when main branch of project is not analyzed
Today, it's possible to analyze a branch of a provisioned project whereas main branch is not analyzed. With SONAR-10221, we now have a clear message in the homepage of the project. But in the projects page, the message for such a project is still confusing: "Project is not analyzed yet.".

## T03

Do not display FOP INFO logs when generating a PDF report
FOP displays logs each time a report is generated, for each page. The FOP logger should be overloaded with something similar to in  FileReportGenerator : Logger logger = Loggers.get("org.apache.fop");
logger.setLevel(LoggerLevel.ERROR);

## T04

Remove the need for Commit Statuses permission
WHY In the past, the permission “Commit Statuses” was required for github applications. After analysis, this is not the case anymore so it makes no point to ask our users to set this permission. WHAT update documentation to remove mentions to this permission (  ) update SonarQube Github settings validation, to not display a warning if this permission is missing. HOW The what is self-explanatory.

## T05

Project creation form validation is buggy
When creating a project, the field validation kicks in for wrong/unexpected formats, but doesn't actually prevent creation. Also, some formats that should be working (according to description) aren't accepted.

## T06

Upgrade analyzers
Changes Replace com.sonarsource.cpp: sonar-cfamily-plugin :6.6.0.13759 by  6.7.0.15300   Replace com.sonarsource.security: sonar-security-csharp-frontend-plugin :8.1.0.675 by  8.2.1.1259 Replace com.sonarsource.security: sonar-security-java-frontend-plugin :8.1.0.675 by  8.2.1.1259 Replace com.sonarsource.security: sonar-security-php-frontend-plugin :8.1.0.675 by  8.2.1.1259 Replace com.sonarsource.security: sonar-security-plugin :8.1.0.675 by  8.2.1.1259 Replace com.sonarsource.security: sonar-security-python-frontend-plugin :8.1.0.675 by  8.2.1.1259 Replace org.sonarsource.css: sonar-css-plugin :1.1.1.1010 by  1.2.0.1325 Replace org.sonarsource.dotnet: sonar-csharp-plugin :8.1.0.13383 by  8.4.0.15306 Replace org.sonarsource.dotnet: sonar-vbnet-plugin :8.1.0.13383 by  8.4.0.15306 Replace org.sonarsource.java: sonar-java-plugin :6.0.0.20538 by  6.1.0.20866   Replace org.sonarsource.javascript: sonar-javascript-plugin :6.1.0.11503 by  6.2.0.12043 Replace org.sonarsource.python: sonar-python-plugin :2.3.0.5351 by  2.5.0.5733

## T07

Projects can be sorted by Security Review Rating
In the Projects page, it should be possible to sort Projects by Security Review Rating. This metric was introduced when we implements the Security Hotspots page/experience and we forgot to update the Projects page accordingly.

## T08

Fix potential vulnerabilities of bundled libraries
Why The Java and JS third-party libraries bundled in the different components of the SonarQube editions can have vulnerabilities. In a few cases the latter may be exploitable by hackers.  Every potential vulnerability should be addressed by being marked as false-positive or by being fixed. What Automate audit of Java and JS dependencies. The vulnerabilities (CWD) referenced in the central repository NVD should be checked on a regular basis. They don't need to be checked on pull requests because they don't relate to the developed changes. Fix the following vulnerabilities: CVE-2020-13934  https://cve.mitre.org/cgi-bin/cvename.cgi?name=2020-13934  by upgrading Tomcat to 8.5.57 or greater CVE-2020-13935  https://cve.mitre.org/cgi-bin/cvename.cgi?name=2020-13935  by upgrading Tomcat to 8.5.57 or greater CVE-2018-19360  https://cve.mitre.org/cgi-bin/cvename.cgi?name=2018-19360  in jackson-core in the commons-codec bundled with Elasticsearch  https://issues.apache.org/jira/browse/CODEC-134 in the http-client bundled with Elasticsearch  https://issues.apache.org/jira/browse/HTTPCLIENT-1803

## T09

Remove no more used Sonar-Version HTTP header
The  Sonar-Version  HTTP header was used by the blue/green feature, which is no more used on SonarQube, and which has never been a public feature. As it's no more used anymore, this header should be removed.

## T10

Github onboarding should be able to search forked project
During the onboarding, users are able to search for repositories in a GH organization. The  default search behaviour is to exclude forked repository . We want repository forked inside an organization to be visible. Solution: add a "fork:true" parameter in the search query.
