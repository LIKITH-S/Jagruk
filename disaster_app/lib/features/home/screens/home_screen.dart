import 'package:flutter/material.dart';

import '../../alerts/screens/alerts_screen.dart';

import '../../safety/screens/safety_screen.dart';

import '../../report/screens/report_screen.dart';

import '../../news/screens/news_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  int currentIndex = 0;

  final screens = [
    const AlertsScreen(),

    const SafetyScreen(),

    const ReportScreen(),

    const NewsScreen(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: screens[currentIndex],

      bottomNavigationBar: BottomNavigationBar(
        currentIndex: currentIndex,

        type: BottomNavigationBarType.fixed,

        onTap: (index) {
          setState(() {
            currentIndex = index;
          });
        },

        items: const [
          BottomNavigationBarItem(icon: Icon(Icons.warning), label: "Alerts"),

          BottomNavigationBarItem(icon: Icon(Icons.security), label: "Safety"),

          BottomNavigationBarItem(icon: Icon(Icons.report), label: "Report"),

          BottomNavigationBarItem(icon: Icon(Icons.newspaper), label: "News"),
        ],
      ),
    );
  }
}
