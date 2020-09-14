""" Database description.

Session is performed by a subject.
Session has multiple blocks.
Block has multiple trials.
Sessions, blocks, and trials have events of certain event type at certain time.
Trial has parameters.

"""

import os
import sys
from sqlalchemy import Column, ForeignKey, Integer, String, Float, DateTime, Time, Enum, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref, reconstructor
from sqlalchemy import create_engine
from sqlalchemy_utils import drop_database
import json 

Base = declarative_base()
 


class Subject(Base):
    __tablename__ = "subject"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(250), nullable=False, unique=True)
    age = Column(Float, nullable=False)
    sex = Column(Enum("male", "female", name="sex_types"))
    handness = Column(Enum("left", "right", name="hand_types"))



class Session(Base):
    __tablename__ = "session"

    id = Column(Integer, primary_key=True)
    subject_id = Column(Integer, ForeignKey("subject.id"), index=True)
    subject = relationship(
        Subject, 
        backref=backref("sessions", uselist=True, cascade="delete,all"))



class EventType(Base):
    __tablename__ = "event_type"

    id = Column(Integer, primary_key=True)
    desc = Column(String(250), nullable=False, unique=True)



class Block(Base):
    __tablename__ = "block"

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("session.id"))
    paramslist = Column(String(10000), nullable=False)
    number = Column(Integer, nullable=False, index=True)  # block number within session
    opto_filename = Column(String(1000), nullable=False)
    odau_filename = Column(String(1000), nullable=False)
    session = relationship(
        Session, 
        backref=backref("blocks", uselist=True, cascade="delete,all"))



class BlockEvent(Base):
    __tablename__ = "block_event"

    id = Column(Integer, primary_key=True)
    block_id = Column(Integer, ForeignKey("block.id"), index=True)
    event_type_id = Column(Integer, ForeignKey("event_type.id"), index=True)
    time = Column(DateTime, nullable=False, index=True)
    event_type = relationship(EventType)
    block = relationship(
        Block, 
        backref=backref("events", uselist=True, cascade="delete,all"))



class Trial(Base):
    __tablename__ = "trial"

    id = Column(Integer, primary_key=True)
    block_id = Column(Integer, ForeignKey("block.id"), index=True)
    paramslist = Column(String(10000), nullable=True)
    number = Column(Integer, nullable=False, index=True)  # trial number within block
    disturbance_mode = Column(Integer, nullable=False, index=True)
    feedback_delay = Column(Float, nullable=False, index=True)
    valid = Column(Boolean, nullable=True, index=True, default=None)  # trial is valid for analysis
    opto_start = Column(Integer)
    opto_stop = Column(Integer)
    odau_start = Column(Integer)
    odau_stop = Column(Integer)
    block = relationship(
        Block, 
        backref=backref("trials", uselist=True, cascade="delete,all"))

    @reconstructor
    def init_on_load(self):
        self.params = json.loads(self.paramslist)



class TrialEvent(Base):
    __tablename__ = "trial_event"

    id = Column(Integer, primary_key=True)
    trial_id = Column(Integer, ForeignKey("trial.id"), index=True)
    event_type_id = Column(Integer, ForeignKey("event_type.id"), index=True)
    time = Column(DateTime, nullable=False, index=True)
    event_type = relationship(EventType)
    trial = relationship(
        Trial, 
        backref=backref("events", uselist=True, cascade="delete,all"))



class SessionEvent(Base):
    __tablename__ = "session_event"

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("session.id"), index=True)
    event_type = Column(Integer, ForeignKey("event_type.id"), index=True)
    time = Column(DateTime, nullable=False, index=True)
    session = relationship(
        Session, 
        backref=backref("trials", uselist=True, cascade="delete,all"))




    
